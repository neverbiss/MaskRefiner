# Generated Person Processing

Scripts for processing generated person images with various SAM models.

## Overview

This directory contains scripts for:
- Processing generated person images with SAM, HQ-SAM, and SAM3
- Combining results from multiple models
- Verifying mask quality

## Scripts

- `process_generated_person_sam1.py` - Process with SAM
- `process_generated_person_hqsam.py` - Process with HQ-SAM
- `process_generated_person_sam3.py` - Process with SAM3
- `process_generated_person_sam3_text.py` - Process with SAM3 text grounding
- `combine_results.py` - Combine results from multiple models
- `run_all.py` - Run all processing scripts
- `verify_6masks.py` - Verify mask quality

## Usage

```bash
# Run all processing
python run_all.py

# Or run individual scripts
python process_generated_person_sam3.py --checkpoint /path/to/sam3.pt
```
