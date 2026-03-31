# Training SegFormer (Segmenter Transformer)

This document describes the training process for the SegFormer model used in FarmTrack Analytics.

## Overview

SegFormer is a transformer-based semantic segmentation model that combines a hierarchical transformer encoder (MiT-B0 for this project) and an efficient MLP decoder. It provides high-resolution segmentation without large computational overhead.

## Training Pipeline

The training script `src/train_segformer.py` utilizes the Hugging Face `transformers` implementation through our `SegformerFarmTrack` adapter.

### Key Components

1. **Backbone**: `nvidia/mit-b0` (Mix Transformer B0) for efficient feature extraction.
2. **MLP Decoder**: A simple MLP head that maps multi-level features from the transformer blocks to a segmentation mask.
3. **Data Pipeline**:
   - `AgVisionDictDataset`: Standard image resizing to 512x512.
   - `FarmTrackDataModule`: Splits Agriculture-Vision dataset (80/20 train/val).
   - Batch size **4** (Due to transformer VRAM requirements on T4).

### Hyperparameters

- **Learning Rate**: `6e-5` (Specific rate fortransformer backbones).
- **Optimizer**: `AdamW`
- **Loss Function**: Combined `DiceLoss` + `BCEWithLogitsLoss`.
- **Precision**: `16-mixed` (Mixed precision training).
- **Epochs**: 15 (Transfomers need more epochs to converge than fine-tuning SAM).

## Output

The final weights are saved to `models/weights/segformer_farmtrack_final.pth`. These weights contain only the state dict for the Segformer model.

## How to Run

```bash
python src/train_segformer.py
```
