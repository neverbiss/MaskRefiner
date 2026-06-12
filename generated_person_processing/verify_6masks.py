#!/usr/bin/env python3
"""Verify mask quality for 6-mask setup."""
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def verify_masks(mask_dir, min_area=100, max_area_ratio=0.9):
    """Verify masks in directory.

    Args:
        mask_dir: Directory containing masks.
        min_area: Minimum mask area.
        max_area_ratio: Maximum area ratio relative to image.

    Returns:
        stats: Dictionary with verification statistics.
    """
    stats = {"total": 0, "valid": 0, "too_small": 0, "too_large": 0, "empty": 0}

    for filename in sorted(os.listdir(mask_dir)):
        if not filename.endswith('.png'):
            continue

        stats["total"] += 1
        mask = np.array(Image.open(os.path.join(mask_dir, filename)))
        mask_binary = mask > 127
        area = mask_binary.sum()
        total_pixels = mask_binary.size

        if area == 0:
            stats["empty"] += 1
        elif area < min_area:
            stats["too_small"] += 1
        elif area / total_pixels > max_area_ratio:
            stats["too_large"] += 1
        else:
            stats["valid"] += 1

    return stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--min_area", type=int, default=100)
    parser.add_argument("--max_area_ratio", type=float, default=0.9)
    args = parser.parse_args()

    stats = verify_masks(args.mask_dir, args.min_area, args.max_area_ratio)

    print(f"Mask Verification Results:")
    print(f"  Total masks: {stats['total']}")
    print(f"  Valid masks: {stats['valid']}")
    print(f"  Too small: {stats['too_small']}")
    print(f"  Too large: {stats['too_large']}")
    print(f"  Empty masks: {stats['empty']}")


if __name__ == "__main__":
    main()
