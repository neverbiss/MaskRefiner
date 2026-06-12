#!/usr/bin/env python3
"""Process generated person images with SAM3."""
import os
import argparse
import numpy as np
from PIL import Image
from maskforge.sam3_refiner import create_sam3_refiner, sam3_refiner


def process_images(image_dir, output_dir, checkpoint, iters=5):
    """Process all images in directory with SAM3."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    sam3_model, processor = create_sam3_refiner(checkpoint)

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

        # Refine with SAM3
        try:
            refined = sam3_refiner(
                image_path, [init_mask], sam3_model, processor, iters=iters
            )

            output_path = os.path.join(output_dir, filename.replace(".jpg", ".png"))
            Image.fromarray(255 * refined[0].astype(np.uint8)).save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--output_dir", default="output/sam3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    process_images(args.image_dir, args.output_dir, args.checkpoint, args.iters)


if __name__ == "__main__":
    main()
