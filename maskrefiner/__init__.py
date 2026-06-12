"""
MaskRefiner - Universal mask refinement for Segment Anything Models.

Iterative mask refinement framework supporting SAM, HQ-SAM, and SAM3
with noise-tolerant prompt generation (points, boxes, and Gaussian mask priors).
"""

__version__ = "1.0.0"

from .sam_refiner import sam_refiner
from .utils import (
    extract_bboxes_expand,
    extract_mask,
    extract_points,
    get_mask_embed,
    prepare_image,
)

__all__ = [
    "sam_refiner",
    "prepare_image",
    "extract_bboxes_expand",
    "extract_points",
    "extract_mask",
    "get_mask_embed",
]
