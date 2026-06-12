"""Demo: SAM3 text-grounding refinement."""
import argparse
import numpy as np
from PIL import Image
from maskforge.sam3_text_grounding_refiner import (
    create_sam3_text_refiner,
    sam3_text_grounding_refiner,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="SAM3 checkpoint path")
    parser.add_argument("--image", default="examples/2007_000256.jpg")
    parser.add_argument("--mask", default="examples/2007_000256_init_mask.png")
    parser.add_argument("--text", default="person", help="Text prompt for grounding")
    parser.add_argument("--iters", type=int, default=3)
    parser.add_argument("--output", default="refined_mask_sam3_text.png")
    args = parser.parse_args()

    # Load model
    sam3_model, processor = create_sam3_text_refiner(args.checkpoint)

    # Load coarse mask
    init_mask = np.asarray(Image.open(args.mask), dtype=np.uint8)
    init_mask = (init_mask > 0).astype(np.uint8)

    # Refine with text grounding
    refined = sam3_text_grounding_refiner(
        args.image,
        [init_mask],
        sam3_model,
        processor,
        text_prompt=args.text,
        iters=args.iters,
    )

    # Save result
    Image.fromarray(255 * refined[0].astype(np.uint8)).save(args.output)
    print(f"Saved refined mask to {args.output}")


if __name__ == "__main__":
    main()
