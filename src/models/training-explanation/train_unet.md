# Training U-Net (Convolutional Encoder-Decoder)

Training process for the PyTorch U-Net model used in FarmTrack Analytics

## Overview

U-Net is a classic convolutional neural network architecture for semantic segmentation. It uses a symmetric encoder-decoder structure with skip connections to preserve high-resolution features.

## Training Pipeline

The training script `src/train_unet.py` implements a standard convolutional approach for agricultural vision tasks.

### Key Components

1. **Encoders**: Hierarchical pooling through 4 convolutional blocks.
2. **Decoders**: Progressive upsampling and concatenation with corresponding encoder features via skip connections.
3. **Data Pipeline**:
   - `AgVisionDictDataset`: Standard image resizing to 512x512.
   - `FarmTrackDataModule`: High-throughput loading.
   - Batch size **8** (Most efficient for T4 GPU/CNN workloads).

### Hyperparameters

- **Learning Rate**: `1e-4` (Standard rate for CNN-based segmentation).
- **Optimizer**: `AdamW`
- **Scheduler**: `CosineAnnealingLR` (Gradual learning rate decay).
- **Loss Function**: Combined `DiceLoss` + `BCEWithLogitsLoss`.
- **Epochs**: 10 (U-Net converges reliably on this dataset).

## Output

The final weights are saved to `models/weights/unet_farmtrack_final.pth`. These weights contain only the state dict for the UNet model.

Model size: 97.9MB

## How to Run

```bash
python src/train_unet.py
```
