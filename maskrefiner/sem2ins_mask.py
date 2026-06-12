"""
Semantic-to-instance mask conversion.

Converts semantic segmentation maps to per-instance binary masks.
"""

import numpy as np
from scipy import ndimage


def sem2ins_masks(semantic_map, ignore_labels=None):
    """Convert semantic segmentation map to instance masks."""
    if ignore_labels is None:
        ignore_labels = [0]

    instance_masks = []
    instance_labels = []

    unique_labels = np.unique(semantic_map)
    for label in unique_labels:
        if label in ignore_labels:
            continue

        class_mask = (semantic_map == label).astype(np.uint8)
        labeled_array, num_features = ndimage.label(class_mask)

        for instance_id in range(1, num_features + 1):
            instance_mask = (labeled_array == instance_id).astype(np.uint8)
            if instance_mask.sum() > 0:
                instance_masks.append(instance_mask)
                instance_labels.append(label)

    return instance_masks, instance_labels


def sem2ins_masks_with_area(semantic_map, min_area=100, ignore_labels=None):
    """Convert semantic segmentation to instance masks with area filtering."""
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
