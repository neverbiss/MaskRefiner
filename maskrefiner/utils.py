"""
Prompt extraction utilities for MaskRefiner.

Provides noise-tolerant prompt generation from coarse masks:
  - Point prompts via geodesic distance transforms
  - Box prompts with optional CEBox (Context-Embedding Box) expansion
  - Gaussian-weighted mask prompts
"""

import FastGeodis
import numpy as np
import torch
import torch.nn.functional as F


def prepare_image(image, transform, device):
    """Apply SAM's ResizeLongestSide transform to an image array."""
    image = transform.apply_image(image)
    image = torch.as_tensor(image, device=device)
    return image.permute(2, 0, 1).contiguous()


def gaussian_2d(shape, gamma_x=1, gamma_y=1):
    """Create a 2D Gaussian kernel."""
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x / (2 * gamma_x * gamma_x) + y * y / (2 * gamma_y * gamma_y)))
    return h


def get_mask_embed(mask, img_embed):
    """Extract a mask embedding by spatial pooling of image features."""
    orig_H, orig_W = mask.shape[:2]
    embed_H, embed_W = img_embed.shape[-2:]
    if orig_H >= orig_W:
        resize_W = int(embed_H * orig_W / orig_H)
        resize_H = embed_H
    else:
        resize_H = int(embed_W * orig_H / orig_W)
        resize_W = embed_W
    mask_resize = F.interpolate(
        mask[None, None].float(), size=(resize_H, resize_W), mode="nearest"
    )
    query_embed = (img_embed[:, :, :resize_H, :resize_W] * mask_resize).sum(
        dim=(-2, -1)
    ) / mask_resize.sum()
    return query_embed, mask_resize


def extract_bboxes_expand(image_embeddings, mask, margin=0):
    """Compute bounding boxes from masks with optional CEBox expansion."""
    ori_h, ori_w = mask.shape[-2:]
    if margin > 0 and ori_h > 0 and ori_w > 0:
        embed_H, embed_W = image_embeddings.shape[-2:]
        if ori_h >= ori_w:
            resize_W = int(embed_H * ori_w / ori_h)
            resize_H = embed_H
        else:
            resize_H = int(embed_W * ori_h / ori_w)
            resize_W = embed_W
        image_embeddings_resize = image_embeddings[:, :, :resize_H, :resize_W]
        image_embeddings_resize = F.interpolate(
            image_embeddings_resize, size=(ori_h, ori_w), mode="bilinear"
        )
        image_embeddings_resize = image_embeddings_resize.permute(0, 2, 3, 1)
        image_embeddings_resize = (
            image_embeddings_resize / image_embeddings_resize.norm(dim=-1, keepdim=True)
        )

    boxes = []
    box_masks = []
    areas = []
    expand_list = []
    for i in range(mask.shape[0]):
        m = mask[i, :, :]
        coord = torch.nonzero(m)
        y_coord, x_coord = coord[:, 0], coord[:, 1]
        try:
            y1, x1 = int(y_coord.min()), int(x_coord.min())
            y2, x2 = int(y_coord.max()), int(x_coord.max())
        except Exception:
            y1, x1 = 0, 0
            y2, x2 = 0, 0

        x1 = max(0, x1)
        y1 = max(0, y1)
        y2 = min(mask.shape[-2] - 1, y2)
        x2 = min(mask.shape[-1] - 1, x2)

        box_h = y2 - y1
        box_w = x2 - x1
        final_x1, final_x2, final_y1, final_y2 = x1, x2, y1, y2
        changed = False

        if box_h > 0 and box_w > 0 and margin > 0 and ori_h > 0 and ori_w > 0:
            steph = min(box_h * 0.1, 10)
            stepw = min(box_w * 0.1, 10)

            query_embed, mask_resize = get_mask_embed(m, image_embeddings)
            query_embed = query_embed / query_embed.norm(dim=-1, keepdim=True)
            sim = image_embeddings_resize @ query_embed.transpose(0, 1)
            sim = sim.squeeze()
            sim = sim > 0.5

            temp_x1 = int(x1 - stepw)
            if temp_x1 > 0 and temp_x1 < x1:
                context_area = (y2 - y1) * (x1 - temp_x1)
                sim_context = sim[y1:y2, temp_x1:x1]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_x1 = temp_x1
                    changed = True

            temp_x2 = int(x2 + stepw)
            if temp_x2 < ori_w and temp_x2 > x2:
                context_area = (y2 - y1) * (temp_x2 - x2)
                sim_context = sim[y1:y2, x2:temp_x2]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_x2 = temp_x2
                    changed = True

            temp_y1 = int(y1 - steph)
            if temp_y1 > 0 and temp_y1 < y1:
                context_area = (y1 - temp_y1) * (x2 - x1)
                sim_context = sim[temp_y1:y1, x1:x2]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_y1 = temp_y1
                    changed = True

            temp_y2 = int(y2 + steph)
            if temp_y2 < ori_h and temp_y2 > y2:
                context_area = (temp_y2 - y2) * (x2 - x1)
                sim_context = sim[y2:temp_y2, x1:x2]
                pos_area = sim_context.sum()
                if pos_area / context_area > margin:
                    final_y2 = temp_y2
                    changed = True

        if changed:
            expand_list.append(1)
        else:
            expand_list.append(0)

        x1, x2, y1, y2 = final_x1, final_x2, final_y1, final_y2
        boxes.append(torch.tensor([x1, y1, x2, y2]))
        box_mask = torch.zeros((m.shape[0], m.shape[1])).to(image_embeddings.device)
        box_mask[y1:y2, x1:x2] = 1
        box_masks.append(box_mask)
        areas.append(1.0 * (x2 - x1) * (y2 - y1))

    boxes = torch.stack(boxes, dim=0).reshape(-1, 4).to(image_embeddings.device)
    box_masks = torch.stack(box_masks, dim=0).to(image_embeddings.device)
    areas = torch.tensor(areas).reshape(-1).to(image_embeddings.device)
    expand_list = torch.tensor(expand_list).reshape(-1).to(image_embeddings.device)
    return boxes, box_masks, areas, expand_list


def extract_points(pred_masks, add_neg=True, use_mask=True, gamma=1.0):
    """Extract point prompts from masks using geodesic distance transforms."""
    device = pred_masks.device
    pred_masks_np = pred_masks.cpu().numpy()
    point_coords = []
    point_labels = []
    gaus_dt_list = []

    for i in range(pred_masks_np.shape[0]):
        mask = pred_masks_np[i]

        # Geodesic distance from boundary
        mask_input = mask[np.newaxis, np.newaxis].astype(np.float32)
        mask_tensor = torch.from_numpy(mask_input).to(device)

        # Compute distance transform
        dist = FastGeodis.geodesic_distance2d(
            mask_tensor,
            v=torch.tensor([1.0, 1.0]).to(device),
            lambd=1.0,
            iters=5,
        )
        dist_np = dist.cpu().numpy().squeeze()

        # Positive point: farthest from boundary inside mask
        dist_inside = dist_np * mask
        if dist_inside.max() > 0:
            idx = np.argmax(dist_inside)
            y_pos, x_pos = np.unravel_index(idx, dist_np.shape)
        else:
            # Fallback: center of mask
            coords = np.argwhere(mask > 0)
            if len(coords) > 0:
                y_pos, x_pos = coords.mean(axis=0).astype(int)
            else:
                y_pos, x_pos = 0, 0

        coords = [[x_pos, y_pos]]
        labels = [1]

        # Negative point: farthest from mask outside
        if add_neg:
            dist_outside = dist_np * (1 - mask)
            if dist_outside.max() > 0:
                idx = np.argmax(dist_outside)
                y_neg, x_neg = np.unravel_index(idx, dist_np.shape)
            else:
                y_neg, x_neg = 0, 0
            coords.append([x_neg, y_neg])
            labels.append(0)

        point_coords.append(coords)
        point_labels.append(labels)

        # Gaussian mask prompt
        if use_mask:
            gaus = gaussian_2d(
                (mask.shape[0], mask.shape[1]),
                gamma_x=gamma * mask.shape[1] / 100,
                gamma_y=gamma * mask.shape[0] / 100,
            )
            gaus = gaus / gaus.max()
            gaus_dt_list.append(gaus)
        else:
            gaus_dt_list.append(np.zeros_like(mask, dtype=np.float32))

    point_coords = torch.tensor(point_coords, dtype=torch.float32).to(device)
    point_labels = torch.tensor(point_labels, dtype=torch.int).to(device)
    gaus_dt = torch.tensor(np.stack(gaus_dt_list), dtype=torch.float32).to(device)

    return point_coords, point_labels, gaus_dt


def extract_mask(
    pred_masks,
    gaus_dt,
    target_size,
    is01=True,
    strength=15,
    device="cuda",
    expand_list=None,
):
    """Prepare Gaussian mask prompts for SAM's mask decoder."""
    mask_inputs = []
    for i in range(pred_masks.shape[0]):
        m = pred_masks[i].float()
        g = gaus_dt[i]

        # Resize to target size
        m_resized = F.interpolate(
            m[None, None], size=target_size, mode="bilinear", align_corners=False
        )
        g_resized = F.interpolate(
            g[None, None], size=target_size, mode="bilinear", align_corners=False
        )

        # Combine mask with Gaussian
        mask_input = m_resized * g_resized * strength
        mask_inputs.append(mask_input)

    mask_inputs = torch.cat(mask_inputs, dim=0)
    return mask_inputs
