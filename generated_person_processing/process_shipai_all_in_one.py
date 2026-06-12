#!/usr/bin/env python3
"""Process shipai (real-world) images with all models."""
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


def process_image(image_path, models, output_dir, args):
    """Process a single image with all models."""
    filename = os.path.basename(image_path)
    output_name = filename.replace(".jpg", ".png").replace(".jpeg", ".png")

    image = np.array(Image.open(image_path).convert("RGB"))
    h, w = image.shape[:2]
    init_mask = np.ones((h, w), dtype=np.uint8)

    results = {}

    # SAM
    try:
        refined, _, _ = sam_refiner(
            image_path, [init_mask], models["sam"], iters=args.iters
        )
        results["sam"] = refined[0]
    except Exception as e:
        print(f"  SAM error: {e}")

    # HQ-SAM
    try:
        refined, _, _ = sam_refiner(
            image_path, [init_mask], models["sam"], iters=args.iters, use_samhq=True
        )
        results["hqsam"] = refined[0]
    except Exception as e:
        print(f"  HQ-SAM error: {e}")

    # SAM3
    try:
        refined = sam3_refiner(
            image_path, [init_mask], models["sam3"], models["processor"], iters=args.iters
        )
        results["sam3"] = refined[0]
    except Exception as e:
        print(f"  SAM3 error: {e}")

    # SAM3 Text
    try:
        refined = sam3_text_grounding_refiner(
            image_path,
            [init_mask],
            models["sam3"],
            models["processor"],
            text_prompt=args.text,
            iters=3,
        )
        results["sam3_text"] = refined[0]
    except Exception as e:
        print(f"  SAM3 Text error: {e}")

    # Save results
    for model_name, mask in results.items():
        model_dir = os.path.join(output_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        Image.fromarray(255 * mask.astype(np.uint8)).save(
            os.path.join(model_dir, output_name)
        )

    return len(results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--output_dir", default="output_shipai")
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--sam3_checkpoint", required=True)
    parser.add_argument("--text", default="person")
    parser.add_argument("--iters", type=int, default=5)
    args = parser.parse_args()

    # Load models
    print("Loading models...")
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda")
    sam3_model, processor = create_sam3_refiner(args.sam3_checkpoint)

    models = {"sam": sam, "sam3": sam3_model, "processor": processor}

    # Process images
    image_files = sorted([
        os.path.join(args.image_dir, f)
        for f in os.listdir(args.image_dir)
        if f.endswith((".jpg", ".png", ".jpeg"))
    ])

    for i, image_path in enumerate(image_files):
        print(f"\nProcessing {i+1}/{len(image_files)}: {os.path.basename(image_path)}")
        n_processed = process_image(image_path, models, args.output_dir, args)
        print(f"  Generated {n_processed} masks")

    print(f"\nDone! Processed {len(image_files)} images")


if __name__ == "__main__":
    main()
