<div align="center">

# MaskRefiner

### Universal Mask Refinement for Segment Anything Models

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-3776AB.svg)](https://www.python.org/)

*Iterative mask refinement through noise-tolerant prompt generation for SAM, HQ-SAM, and SAM3.*

[Getting Started](#getting-started) · [Usage](#usage) · [Architecture](#architecture)

</div>

---

## Overview

**MaskRefiner** is a universal mask refinement framework that improves coarse segmentation masks by iteratively generating noise-tolerant prompts and feeding them back into Segment Anything Models. It supports three SAM backends:

| Backend | Mode | Key Feature |
|---------|------|-------------|
| **SAM** (ViT-H) | Instance | Standard prompt-based refinement |
| **HQ-SAM** (ViT-H) | Instance | High-quality mask decoder with early-layer features |
| **SAM3** | Instance / Text Grounding | Open-vocabulary text prompts + geometric prompts |

### Key Features

- **Three prompt types**: Point prompts (geodesic distance), box prompts (CEBox expansion), and Gaussian mask priors
- **CEBox (Context-Embedding Box)**: Intelligent bounding box expansion via image embedding cosine similarity
- **Multi-backend support**: Works with SAM, HQ-SAM, and SAM3 out of the box
- **IoU Adaption**: Optional training to align SAM's IoU predictions with actual mask quality
- **DDP compatible**: Supports distributed data parallel for batch processing

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌───────────────┐
│  Coarse Mask │─────▶│  Prompt Extractor │─────▶│  SAM Decoder  │
└──────┬──────┘     │                  │     └───────┬───────┘
       ▲            │  • Point (geodesic) │           │
       │            │  • Box   (CEBox)   │           │
       └────────────│  • Mask  (Gaussian)│◀──────────┘
        iteration   └──────────────────┘
```

## Getting Started

```bash
git clone https://github.com/neverbiss/MaskRefiner.git
cd MaskRefiner
pip install -e .
```

## Usage

```python
from maskrefiner import sam_refiner

refined_masks, ious, candidates = sam_refiner(
    image_path="image.png",
    coarse_masks=[mask1, mask2],
    sam=model,
    iters=5,
)
```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
