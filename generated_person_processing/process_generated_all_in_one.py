#!/usr/bin/env python3
"""Process generated person images with all models in one script."""
import os
import argparse
import numpy as np
from PIL import Image
import torch
from segment_anything import sam_model_registry
from maskforge import sam_refiner
from maskforge.sam3_refiner import create_sam3_refiner, sam3_refiner
from maskforge.sam3_text_grounding_refiner import (
    create_sam3_text_refiner,
    sam3_text_grounding_refiner,
)


def process_with_sam(image_path, sam, iters=5):
    """Process single image with SAM."""
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    init_mask = np.ones((h, w), dtype=np.uint8)

    refined, ious, _ = sam_refiner(image_path, [init_mask], sam, iters=iters)
    return refined[0]


def process_with_hqsam(image_path, sam, iters=5):
    """Process single image with HQ-SAM."""
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    init_mask = np.ones((h, w), dtype=np.uint8)

    refined, ious, _ = sam_refiner(
        image_path, [init_mask], sam, iters=iters, use_samhq=True
    )
    return refined[0]


def process_with_sam3(image_path, sam3_model, processor, iters=5):
    """Process single image with SAM3."""
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    init_mask = np.ones((h, w), dtype=np.uint8)

    refined = sam3_refiner(
        image_path, [init_mask], sam3_model, processor, iters=iters
    )
    return refined[0]


def process_with_sam3_text(
    image_path, sam3_model, processor, text_prompt="person", iters=3
):
    """Process single image with SAM3 text grounding."""
    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    init_mask = np.ones((h, w), dtype=np.uint8)

    refined = sam3_text_grounding_refiner(
        image_path,
        [init_mask],
        sam3_model,
        processor,
        text_prompt=text_prompt,
        iters=iters,
    )
    return refined[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", default="generated_images")
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam3_checkpoint", required=True)
    parser.add_argument("--text", default="person")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    # Create output directories
    for model in ["sam", "hqsam", "sam3", "sam3_text"]:
        os.makedirs(os.path.join(args.output_dir, model), exist_ok=True)

    # Load models
    print("Loading SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda")

    print("Loading SAM3...")
    sam3_model, processor = create_sam3_refiner(args.sam3_checkpoint)

    # Process images
    image_files = sorted([
        f for f in os.listdir(args.image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    for i, filename in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {filename}")
        image_path = os.path.join(args.image_dir, filename)
        output_name = filename.replace(".jpg", ".png").replace(".jpeg", ".png")

        # SAM
        try:
            mask = process_with_sam(image_path, sam, args.iters)
            Image.fromarray(255 * mask.astype(np.uint8)).save(
                os.path.join(args.output_dir, "sam", output_name)
            )
        except Exception as e:
            print(f"  SAM error: {e}")

        # HQ-SAM
        try:
            mask = process_with_hqsam(image_path, sam, args.iters)
            Image.fromarray(255 * mask.astype(np.uint8)).save(
                os.path.join(args.output_dir, "hqsam", output_name)
            )
        except Exception as e:
            print(f"  HQ-SAM error: {e}")

        # SAM3
        try:
            mask = process_with_sam3(image_path, sam3_model, processor, args.iters)
            Image.fromarray(255 * mask.astype(np.uint8)).save(
                os.path.join(args.output_dir, "sam3", output_name)
            )
        except Exception as e:
            print(f"  SAM3 error: {e}")

        # SAM3 Text
        try:
            mask = process_with_sam3_text(
                image_path, sam3_model, processor, args.text, 3
            )
            Image.fromarray(255 * mask.astype(np.uint8)).save(
                os.path.join(args.output_dir, "sam3_text", output_name)
            )
        except Exception as e:
            print(f"  SAM3 Text error: {e}")

    print(f"\nDone! Processed {len(image_files)} images")


if __name__ == "__main__":
    main()
