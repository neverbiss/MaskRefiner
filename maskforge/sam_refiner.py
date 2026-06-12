"""
SAM-based iterative mask refinement.

Supports SAM (ViT-H) and HQ-SAM backends. The refiner generates
noise-tolerant prompts (points, boxes, Gaussian mask priors) from
coarse masks and iteratively refines them through SAM's decoder.
"""

import cv2
import numpy as np
import torch
from segment_anything.utils.transforms import ResizeLongestSide

from .utils import extract_bboxes_expand, extract_mask, extract_points, prepare_image


def sam_input_prepare(
    image,
    pred_masks,
    image_embeddings=None,
    resize_transform=None,
    use_point=True,
    use_box=True,
    use_mask=True,
    add_neg=True,
    margin=0.0,
    gamma=1.0,
    strength=15,
):
    """Prepare SAM decoder inputs from predicted masks.

    Extracts point, box, and mask prompts from the current predictions
    and formats them for SAM's mask decoder.

    Args:
        image: (3, H, W) preprocessed image tensor.
        pred_masks: (N, H, W) binary predicted masks.
        image_embeddings: (1, C, h, w) image encoder features.
        resize_transform: ``ResizeLongestSide`` instance for coordinate transforms.
        use_point: Whether to include point prompts.
        use_box: Whether to include box prompts.
        use_mask: Whether to include mask prompts.
        add_neg: Whether to add negative point prompts.
        margin: CEBox expansion threshold for box prompts.
        gamma: Gaussian spread parameter for mask prompts.
        strength: Amplitude scaling for mask prompts.

    Returns:
        input_dict: Dictionary of decoder inputs.
        point_coords: Raw point coordinates (before resize).
    """
    ori_size = pred_masks.shape[-2:]
    input_dict = {
        "image": image,
        "original_size": ori_size,
    }

    target_size = image.shape[1:]
    expand_list = torch.zeros((len(pred_masks))).to(image.device)
    if use_box:
        bboxes, box_masks, areas, expand_list = extract_bboxes_expand(
            image_embeddings, pred_masks, margin=margin
        )
        input_dict["boxes"] = resize_transform.apply_boxes_torch(bboxes, ori_size)

    point_coords, point_labels, gaus_dt = extract_points(
        pred_masks, add_neg=add_neg, use_mask=use_mask, gamma=gamma
    )
    if use_point:
        input_dict["point_coords"] = resize_transform.apply_coords_torch(
            point_coords, ori_size
        )
        input_dict["point_labels"] = point_labels

    if use_mask:
        input_dict["mask_inputs"] = extract_mask(
            pred_masks,
            gaus_dt,
            target_size,
            is01=True,
            strength=strength,
            device=image.device,
            expand_list=expand_list,
        )

    return input_dict, point_coords


def sam_refiner(
    image_path,
    coarse_masks,
    sam,
    resize_transform=None,
    use_point=True,
    use_box=True,
    use_mask=True,
    add_neg=True,
    iters=5,
    margin=0.0,
    gamma=4.0,
    strength=30,
    use_samhq=False,
    ddp=False,
    is_train=False,
):
    """Iteratively refine coarse masks using SAM.

    Given an image and one or more coarse binary masks, generates
    noise-tolerant prompts and feeds them through SAM's decoder for
    a configurable number of iterations.

    Args:
        image_path: Path to the input image.
        coarse_masks: List of (H, W) numpy arrays or a single (N, H, W) array.
        sam: SAM model instance.
        resize_transform: Optional ``ResizeLongestSide`` (auto-created if None).
        use_point: Use point prompts.
        use_box: Use box prompts.
        use_mask: Use mask prompts.
        add_neg: Include negative point prompts.
        iters: Number of refinement iterations.
        margin: CEBox expansion threshold (0 = disabled).
        gamma: Gaussian spread for mask prompt generation.
        strength: Amplitude scaling for mask prompts.
        use_samhq: Use HQ-SAM mode (requires HQ-SAM checkpoint).
        ddp: Whether model is wrapped in DistributedDataParallel.
        is_train: If True, returns raw multi-mask outputs for training.

    Returns:
        refined_masks: (N, H, W) uint8 refined binary masks.
        sam_ious: IoU predictions from SAM.
        sam_masks3: All 3 candidate masks from the last iteration.
    """
    if isinstance(coarse_masks, list):
        coarse_masks = np.stack(coarse_masks, axis=0)

    if len(coarse_masks.shape) == 2:
        coarse_masks = coarse_masks[None:,]
    coarse_masks = torch.tensor(coarse_masks, dtype=torch.uint8).to(sam.device)
    assert (
        len(coarse_masks.shape) == 3
    ), f"coarse mask dim must be (n, h, w), but got {coarse_masks.shape}"

    if resize_transform is None:
        resize_transform = ResizeLongestSide(sam.image_encoder.img_size)

    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = [prepare_image(image, resize_transform, sam.device)]

    with torch.no_grad():
        if ddp:
            input_images = torch.stack([sam.module.preprocess(x) for x in image], dim=0)
            if not use_samhq:
                image_embeddings = sam.module.image_encoder(input_images)
            else:
                image_embeddings, interm_embeddings = sam.module.image_encoder(
                    input_images
                )
                interm_embeddings = interm_embeddings[0]
        else:
            input_images = torch.stack([sam.preprocess(x) for x in image], dim=0)
            if not use_samhq:
                image_embeddings = sam.image_encoder(input_images)
            else:
                image_embeddings, interm_embeddings = sam.image_encoder(input_images)
                interm_embeddings = interm_embeddings[0]

    sam_masks_list = None
    for i in range(iters):
        if i == 0:
            pred_mask_list = coarse_masks
        else:
            pred_mask_list = sam_masks_list.to(torch.uint8)

        input_dict, point_coords = sam_input_prepare(
            image[0],
            pred_mask_list,
            image_embeddings,
            resize_transform,
            use_point=use_point,
            use_box=use_box,
            use_mask=use_mask,
            add_neg=add_neg,
            margin=margin,
            gamma=gamma,
            strength=strength,
        )

        sam_input = [input_dict]

        if not is_train:
            with torch.no_grad():
                if ddp:
                    if not use_samhq:
                        sam_output = sam.module.forward_with_image_embeddings(
                            image_embeddings, sam_input, multimask_output=True
                        )[0]
                    else:
                        sam_output = sam.module.forward_with_image_embeddings(
                            image_embeddings,
                            interm_embeddings,
                            sam_input,
                            multimask_output=True,
                        )[0]
                else:
                    if not use_samhq:
                        sam_output = sam.forward_with_image_embeddings(
                            image_embeddings, sam_input, multimask_output=True
                        )[0]
                    else:
                        sam_output = sam.forward_with_image_embeddings(
                            image_embeddings,
                            interm_embeddings,
                            sam_input,
                            multimask_output=True,
                        )[0]
        else:
            if ddp:
                if not use_samhq:
                    sam_output = sam.module.forward_with_image_embeddings(
                        image_embeddings, sam_input, multimask_output=True
                    )[0]
                else:
                    sam_output = sam.module.forward_with_image_embeddings(
                        image_embeddings,
                        interm_embeddings,
                        sam_input,
                        multimask_output=True,
                    )[0]
            else:
                if not use_samhq:
                    sam_output = sam.forward_with_image_embeddings(
                        image_embeddings, sam_input, multimask_output=True
                    )[0]
                else:
                    sam_output = sam.forward_with_image_embeddings(
                        image_embeddings,
                        interm_embeddings,
                        sam_input,
                        multimask_output=True,
                    )[0]

        # Select best mask based on IoU prediction
        sam_masks = sam_output["masks"]  # (N, 3, H, W)
        sam_ious = sam_output["iou_predictions"]  # (N, 3)
        sam_masks3 = sam_masks

        # Choose the mask with highest predicted IoU
        best_idx = sam_ious.argmax(dim=-1)  # (N,)
        sam_masks = sam_masks[
            torch.arange(sam_masks.shape[0]), best_idx
        ]  # (N, H, W)
        sam_ious = sam_ious[
            torch.arange(sam_ious.shape[0]), best_idx
        ]  # (N,)

        # Threshold
        sam_masks_list = (sam_masks > 0.0).to(torch.uint8)

    return sam_masks_list.cpu().numpy(), sam_ious.cpu().numpy(), sam_masks3
