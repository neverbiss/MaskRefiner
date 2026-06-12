"""
Semantic-to-instance mask conversion.

Converts semantic segmentation maps to per-instance binary masks.
"""

import numpy as np
from scipy import ndimage


def sem2ins_masks(semantic_map, ignore_labels=None):
    """Convert semantic segmentation map to instance masks.

    Args:
        semantic_map: (H, W) array with class labels.
        ignore_labels: List of labels to ignore (e.g., background).

    Returns:
        instance_masks: List of (H, W) binary masks.
        instance_labels: List of class labels for each instance.
    """
    if ignore_labels is None:
        ignore_labels = [0]  # Default: ignore background

    instance_masks = []
    instance_labels = []

    unique_labels = np.unique(semantic_map)
    for label in unique_labels:
        if label in ignore_labels:
            continue

        # Binary mask for this class
        class_mask = (semantic_map == label).astype(np.uint8)

        # Connected components
        labeled_array, num_features = ndimage.label(class_mask)

        for instance_id in range(1, num_features + 1):
            instance_mask = (labeled_array == instance_id).astype(np.uint8)
            if instance_mask.sum() > 0:
                instance_masks.append(instance_mask)
                instance_labels.append(label)

    return instance_masks, instance_labels


def sem2ins_masks_with_area(
    semantic_map, min_area=100, ignore_labels=None
):
    """Convert semantic segmentation to instance masks with area filtering.

    Args:
        semantic_map: (H, W) array with class labels.
        min_area: Minimum area threshold for instances.
        ignore_labels: List of labels to ignore.

    Returns:
        instance_masks: List of (H, W) binary masks.
        instance_labels: List of class labels.
        instance_areas: List of instance areas.
    """
    if ignore_labels is None:
        ignore_labels = [0]

    instance_masks = []
    instance_labels = []
    instance_areas = []

    unique_labels = np.unique(semantic_map)
    for label in unique_labels:
        if label in ignore_labels:
            continue

        class_mask = (semantic_map == label).astype(np.uint8)
        labeled_array, num_features = ndimage.label(class_mask)

        for instance_id in range(1, num_features + 1):
            instance_mask = (labeled_array == instance_id).astype(np.uint8)
            area = instance_mask.sum()

            if area >= min_area:
                instance_masks.append(instance_mask)
                instance_labels.append(label)
                instance_areas.append(area)

    return instance_masks, instance_labels, instance_areas
