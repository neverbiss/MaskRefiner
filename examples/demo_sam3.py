"""Demo: SAM3 instance mask refinement."""
import argparse
import numpy as np
from PIL import Image
from maskforge.sam3_refiner import create_sam3_refiner, sam3_refiner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint path")
    parser.add_argument("--image", default="examples/2007_000256.jpg")
    parser.add_argument("--mask", default="examples/2007_000256_init_mask.png")
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--output", default="refined_mask_sam3.png")
    args = parser.parse_args()

    # Load model
    sam3_model, processor = create_sam3_refiner(args.checkpoint)

    # Load coarse mask
    init_mask = np.asarray(Image.open(args.mask), dtype=np.uint8)
    init_mask = (init_mask > 0).astype(np.uint8)

    # Refine
    refined = sam3_refiner(
        args.image, [init_mask], sam3_model, processor, iters=args.iters
    )

    # Save result
    Image.fromarray(255 * refined[0].astype(np.uint8)).save(args.output)
    print(f"Saved refined mask to {args.output}")


if __name__ == "__main__":
    main()
