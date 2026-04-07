# Training SAM (Segment Anything Model)

Training and fine-tuning process for the Segment Anything Model (SAM) adapter used in FarmTrack Analytics

## Overview

SAM is a foundation model for image segmentation. In this project, we use an adapter (`SAMFarmTrack`) that wraps `facebook/sam-vit-base`. To specialize the model for agricultural tracks without requiring full retraining, we freeze the heavy vision and prompt encoders and only fine-tune the **Mask Decoder**.

## Training Pipeline

The training script `src/train_sam.py` implements a high-performance pipeline optimized for NVIDIA T4 GPUs.

### Key Components

1. **Frozen Encoders**: 
   - `vision_encoder`: Frozen to preserve general image features.
   - `prompt_encoder`: Frozen as we use fixed point prompts.
2. **Trainable Decoder**: Only the mask decoder parameters are updated to learn the specific textures and spatial features of agricultural "planter skip" patterns.
3. **Data Pipeline**:
   - Uses `AgVisionSAMDataset` which pre-processes images using `SamProcessor` on the CPU.
   - Optimized with `num_workers=4`, `pin_memory=True`, and `persistent_workers=True`.
   - Batch size is set to **16** to maximize 16GB VRAM utilization.

### Hyperparameters

- **Learning Rate**: `1e-5` (Lower rate for fine-tuning a pre-trained decoder).
- **Optimizer**: `AdamW`
- **Loss Function**: Combined `DiceLoss` + `BCEWithLogitsLoss`.
- **Precision**: `16-mixed` (Automatic Mixed Precision for T4 speedup).
- **Epochs**: 5 (SAM converges quickly during fine-tuning).

## Output

The final weights are saved to `models/weights/sam_farmtrack_final.pth`. These weights contain only the state dict for the adapter, which includes the fine-tuned mask decoder.

## How to Run

```bash
python src/train_sam.py
```
