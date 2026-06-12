#!/usr/bin/env python3
"""Process generated person images with SAM3 text-only grounding."""
import os
import argparse
import numpy as np
from PIL import Image
from maskforge.sam3_text_grounding_refiner import (
    create_sam3_text_refiner,
    sam3_text_grounding_refiner,
)


def process_images(image_dir, output_dir, checkpoint, text_prompt="person"):
    """Process all images with text-only grounding (no initial mask)."""
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    sam3_model, processor = create_sam3_text_refiner(checkpoint)

    # Process each image
    image_files = sorted([
        f for f in os.listdir(image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    for i, filename in enumerate(image_files):
        print(f"Processing {i+1}/{len(image_files)}: {filename}")

        image_path = os.path.join(image_dir, filename)
        image = np.array(Image.open(image_path).convert("RGB"))

        # No initial mask - use text grounding only
        init_mask = np.zeros(image.shape[:2], dtype=np.uint8)

        # Refine with SAM3 text grounding
        try:
            refined = sam3_text_grounding_refiner(
                image_path,
                [init_mask],
                sam3_model,
                processor,
                text_prompt=text_prompt,
                iters=1,
            )

            output_path = os.path.join(output_dir, filename.replace(".jpg", ".png"))
            Image.fromarray(255 * refined[0].astype(np.uint8)).save(output_path)
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Processed {len(image_files)} images")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--output_dir", default="output/sam3_text_only")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", default="person")
    args = parser.parse_args()

    process_images(args.image_dir, args.output_dir, args.checkpoint, args.text)


if __name__ == "__main__":
    main()
