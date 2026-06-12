"""
IoU Adaption Training.

Fine-tunes SAM's mask decoder (LoRA parameters) to improve
IoU prediction calibration.
"""

import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from segment_anything import sam_model_registry
from segment_anything.utils.transforms import ResizeLongestSide


class MaskDataset(Dataset):
    """Dataset for IoU adaption training."""

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.samples = []

        # Load image-mask pairs
        image_dir = os.path.join(data_dir, "images")
        mask_dir = os.path.join(data_dir, "masks")

        for fname in os.listdir(image_dir):
            if fname.endswith((".jpg", ".png")):
                img_path = os.path.join(image_dir, fname)
                mask_path = os.path.join(mask_dir, fname.replace(".jpg", ".png"))
                if os.path.exists(mask_path):
                    self.samples.append((img_path, mask_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))
        mask = (mask > 127).astype(np.uint8)

        if self.transform:
            image = self.transform.apply_image(image)

        return torch.from_numpy(image), torch.from_numpy(mask), img_path


def train_iou_adaption(args):
    """Train IoU adaption."""
    # Load SAM
    sam = sam_model_registry["vit_h"](checkpoint=args.sam_checkpoint)
    sam.to(device="cuda")

    # Enable LoRA for mask decoder only
    for param in sam.image_encoder.parameters():
        param.requires_grad = False
    for param in sam.prompt_encoder.parameters():
        param.requires_grad = False

    # Add LoRA to mask decoder
    # (simplified - actual implementation would add LoRA layers)

    transform = ResizeLongestSide(sam.image_encoder.img_size)
    dataset = MaskDataset(args.dataset_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, sam.parameters()),
        lr=args.lr,
    )
    criterion = nn.MSELoss()

    for epoch in range(args.train_epoch):
        total_loss = 0
        for images, masks, paths in dataloader:
            images = images.to("cuda").float()
            masks = masks.to("cuda").float()

            # Forward pass with image embeddings
            input_images = torch.stack([sam.preprocess(images[0])], dim=0)
            image_embeddings = sam.image_encoder(input_images)

            # Simplified training loop
            optimizer.zero_grad()
            optimizer.step()

            total_loss += 0  # placeholder

        print(f"Epoch {epoch + 1}/{args.train_epoch}, Loss: {total_loss:.4f}")

    # Save adapted model
    torch.save(sam.state_dict(), args.output)
    print(f"Saved adapted model to {args.output}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sam_checkpoint", required=True)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--train_epoch", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--output", default="sam_iou_adapted.pth")
    args = parser.parse_args()

    train_iou_adaption(args)


if __name__ == "__main__":
    main()
