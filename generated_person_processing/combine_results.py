#!/usr/bin/env python3
"""Combine results from multiple processing scripts."""
import os
import argparse
import numpy as np
from PIL import Image
from pathlib import Path


def combine_masks(input_dirs, output_dir, method="union"):
    """Combine masks from multiple directories.

    Args:
        input_dirs: List of directories containing masks.
        output_dir: Output directory.
        method: Combination method ('union' or 'intersection').
    """
    os.makedirs(output_dir, exist_ok=True)

    # Get all mask filenames
    all_files = set()
    for d in input_dirs:
        files = [f for f in os.listdir(d) if f.endswith('.png')]
        all_files.update(files)

    for filename in sorted(all_files):
        masks = []
        for d in input_dirs:
            path = os.path.join(d, filename)
            if os.path.exists(path):
                mask = np.array(Image.open(path))
                masks.append(mask > 127)

        if not masks:
            continue

        if method == "union":
            combined = np.logical_or.reduce(masks).astype(np.uint8) * 255
        else:  # intersection
            combined = np.logical_and.reduce(masks).astype(np.uint8) * 255

        output_path = os.path.join(output_dir, filename)
        Image.fromarray(combined).save(output_path)

    print(f"Combined {len(all_files)} masks to {output_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dirs", nargs="+", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--method", choices=["union", "intersection"], default="union")
    args = parser.parse_args()

    combine_masks(args.input_dirs, args.output_dir, args.method)


if __name__ == "__main__":
    main()
