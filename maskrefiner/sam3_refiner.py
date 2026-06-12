"""
SAM3-based iterative mask refinement.

Uses SAM3's instance interactivity mode for refinement with
geometric prompts (points, boxes) extracted from coarse masks.
"""

import cv2
import numpy as np
import torch

from .utils import extract_points


def create_sam3_refiner(checkpoint_path, device="cuda"):
    """Create SAM3 model and processor for instance refinement."""
    from sam3.model import build_sam3_image_model

    sam3_model = build_sam3_image_model(checkpoint_path, device=device)
    processor = sam3_model["processor"]
    return sam3_model, processor


def sam3_refiner(
    image_path,
    coarse_masks,
    sam3_model,
    processor,
    iters=5,
    margin=0.0,
    gamma=4.0,
):
    """Iteratively refine coarse masks using SAM3."""
    if isinstance(coarse_masks, list):
        coarse_masks = np.stack(coarse_masks, axis=0)

    if len(coarse_masks.shape) == 2:
        coarse_masks = coarse_masks[None, :]

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processor.set_image(image)

    refined_masks = []
    for mask_idx in range(coarse_masks.shape[0]):
        current_mask = torch.tensor(
            coarse_masks[mask_idx:mask_idx + 1], dtype=torch.uint8
        )

        for iteration in range(iters):
            # Extract prompts
            point_coords, point_labels, _ = extract_points(
                current_mask, add_neg=True, use_mask=False, gamma=gamma
            )

            # Convert to numpy for SAM3
            points_np = point_coords.cpu().numpy()[0]
            labels_np = point_labels.cpu().numpy()[0]

            # Run SAM3 prediction
            processor.add_geometric_prompt(
                points=points_np.tolist(),
                labels=labels_np.tolist(),
            )
            result = processor.predict_inst()

            if result is not None and "masks" in result:
                # Take best mask
                masks = result["masks"]
                ious = result.get("iou_predictions", [1.0] * len(masks))
                best_idx = np.argmax(ious)
                current_mask = torch.tensor(
                    masks[best_idx:best_idx + 1], dtype=torch.uint8
                )
            else:
                break

        refined_masks.append(current_mask.cpu().numpy()[0])

    return refined_masks
