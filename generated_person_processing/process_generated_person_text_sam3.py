#!/usr/bin/env python3
"""Process generated person images with text-guided SAM3."""
import os
import argparse
import numpy as np
from PIL import Image
from maskforge.sam3_text_grounding_refiner import (
    create_sam3_text_refiner,
    sam3_text_grounding_refiner,
)


def process_images(
    image_dir, output_dir, checkpoint, text_prompts, iters=3
):
    """Process images with multiple text prompts."""
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

        # Try each text prompt
        for text_prompt in text_prompts:
            # Create initial mask
            h, w = image.shape[:2]
            init_mask = np.ones((h, w), dtype=np.uint8)

            try:
                refined = sam3_text_grounding_refiner(
                    image_path,
                    [init_mask],
                    sam3_model,
                    processor,
                    text_prompt=text_prompt,
                    iters=iters,
                )

                # Save with text prompt in filename
                name, ext = os.path.splitext(filename)
                output_name = f"{name}_{text_prompt.replace(' ', '_')}{ext}"
                output_path = os.path.join(output_dir, output_name)
                Image.fromarray(255 * refined[0].astype(np.uint8)).save(output_path)
            except Exception as e:
                print(f"  Error with '{text_prompt}': {e}")

    print(f"Processed {len(image_files)} images with {len(text_prompts)} prompts")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--output_dir", default="output/text_sam3")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--text", nargs="+", default=["person", "human", "figure"])
    parser.add_argument("--iters", type=int, default=3)
    args = parser.parse_args()

    process_images(args.image_dir, args.output_dir, args.checkpoint, args.text, args.iters)


if __name__ == "__main__":
    main()
