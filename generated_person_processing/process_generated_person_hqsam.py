#!/usr/bin/env python3
"""Process generated person images with HQ-SAM."""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry
from maskforge import sam_refiner


def process_images(image_dir, output_dir, checkpoint, iters=5):
    """Process all images in directory with HQ-SAM."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
    sam.to(device="cuda")

    # Process each image
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    for i, filename in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {filename}")

        image_path = os.path.join(image_dir, filename)
        image = np.array(Image.open(image_path).convert("RGB"))

        # Create initial mask
        h, w = image.shape[:2]
        init_mask = np.ones((h, w), dtype=np.uint8)

        # Refine with HQ-SAM
        try:
            refined, ious, _ = sam_refiner(
                image_path, [init_mask], sam, iters=iters, use_samhq=True
            )

            output_path = os.path.join(output_dir, filename.replace(".jpg", ".png"))
            Image.fromarray(255 * refined[0].astype(np.uint8)).save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--output_dir", default="output/hqsam")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    process_images(args.image_dir, args.output_dir, args.checkpoint, args.iters)


if __name__ == "__main__":
    main()
