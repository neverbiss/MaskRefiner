"""
SAM3 text-grounding refinement.

Uses SAM3's text grounding capability to refine masks with
open-vocabulary text prompts combined with geometric prompts.
"""

import cv2
import numpy as np
import torch

from .utils import extract_bboxes_expand, extract_points


def create_sam3_text_refiner(checkpoint_path, device="cuda"):
    """Create SAM3 model for text-grounding refinement.

    Args:
        checkpoint_path: Path to SAM3 checkpoint.
        device: Target device.

    Returns:
        sam3_model: SAM3 model instance.
        processor: Sam3Processor instance.
    """
    from sam3.model import build_sam3_image_model

    sam3_model = build_sam3_image_model(checkpoint_path, device=device)
    processor = sam3_model["processor"]
    return sam3_model, processor


def sam3_text_grounding_refiner(
    image_path,
    coarse_masks,
    sam3_model,
    processor,
    text_prompt,
    confidence_threshold=0.3,
    iters=3,
):
    """Refine masks using SAM3 text grounding.

    Combines text-grounded segmentation with geometric prompt refinement
    to improve mask quality.

    Args:
        image_path: Path to the input image.
        coarse_masks: List of (H, W) numpy arrays.
        sam3_model: SAM3 model dict.
        processor: Sam3Processor instance.
        text_prompt: Text description for grounding.
        confidence_threshold: Minimum confidence for text-grounded masks.
        iters: Number of refinement iterations.

    Returns:
        refined_masks: List of (H, W) refined binary masks.
    """
    if isinstance(coarse_masks, list):
        coarse_masks = np.stack(coarse_masks, axis=0)

    if len(coarse_masks.shape) == 2:
        coarse_masks = coarse_masks[None, :]

    # Load and preprocess image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    processor.set_image(image)

    # First: text-grounded segmentation
    processor.set_text_prompt(text_prompt)
    text_result = processor.forward_grounding()

    text_masks = []
    if text_result is not None and "masks" in text_result:
        for i, (mask, score) in enumerate(
            zip(text_result["masks"], text_result.get("scores", [1.0] * len(text_result["masks"])))
        ):
            if score >= confidence_threshold:
                text_masks.append(mask)

    # Combine coarse masks with text-grounded masks
    all_masks = list(coarse_masks) + text_masks

    # Refine combined masks with geometric prompts
    refined_masks = []
    for mask in all_masks:
        current_mask = torch.tensor(mask[np.newaxis], dtype=torch.uint8)

        for iteration in range(iters):
            # Extract geometric prompts
            point_coords, point_labels, _ = extract_points(
                current_mask, add_neg=True, use_mask=False
            )

            points_np = point_coords.cpu().numpy()[0]
            labels_np = point_labels.cpu().numpy()[0]

            # SAM3 instance prediction
            processor.add_geometric_prompt(
                points=points_np.tolist(),
                labels=labels_np.tolist(),
            )
            result = processor.predict_inst()

            if result is not None and "masks" in result:
                masks = result["masks"]
                ious = result.get("iou_predictions", [1.0] * len(masks))
                best_idx = np.argmax(ious)
                current_mask = torch.tensor(
                    masks[best_idx : best_idx + 1], dtype=torch.uint8
                )
            else:
                break

        refined_masks.append(current_mask.cpu().numpy()[0])

    return refined_masks
