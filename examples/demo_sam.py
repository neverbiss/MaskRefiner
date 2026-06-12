"""Demo: SAM mask refinement."""
import argparse
import numpy as np
from PIL import Image
from segment_anything import sam_model_registry
from maskforge import sam_refiner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="SAM checkpoint path")
    parser.add_argument("--image", default="examples/2007_000256.jpg")
    parser.add_argument("--mask", default="examples/2007_000256_init_mask.png")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", default="refined_mask.png")
    args = parser.parse_args()

    # Load model
    sam = sam_model_registry["vit_h"](checkpoint=args.checkpoint)
    sam.to(device="cuda")

    # Load coarse mask
    init_mask = np.asarray(Image.open(args.mask), dtype=np.uint8)
    init_mask = (init_mask > 0).astype(np.uint8)

    # Refine
    refined, ious, _ = sam_refiner(
        args.image, [init_mask], sam, iters=args.iters
    )

    # Save result
    Image.fromarray(255 * refined[0].astype(np.uint8)).save(args.output)
    print(f"Saved refined mask to {args.output}")
    print(f"IoU predictions: {ious}")


if __name__ == "__main__":
    main()
