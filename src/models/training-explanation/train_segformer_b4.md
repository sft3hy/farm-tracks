# Training SegFormer B4 (A100 Full-Utilization)

Training process for the high-performance SegFormer-B4 model used in FarmTrack Analytics

## Overview

This is an upgraded SegFormer variant using the **MiT-B4** backbone, trained on a Google Colab A100 GPU with full-utilization optimizations. Compared to the baseline SegFormer-B0, B4 has ~60M parameters (vs ~3.7M), deeper transformer blocks, and significantly better feature extraction capacity.

## Training Pipeline

The training script `colab_a100_segformer/train_segformer_a100.py` runs a standalone PyTorch Lightning pipeline tuned specifically for the A100's Tensor Cores.

### Key Components

1. **Backbone**: `nvidia/mit-b4` (Mix Transformer B4) — 4× deeper than B0 with more attention heads per stage.
2. **MLP Decoder**: Same lightweight MLP head as SegFormer-B0, but receiving richer multi-level features.
3. **Data Pipeline**:
   - `AgVisionDictDataset`: Standard image resizing to 512×512.
   - `FarmTrackDataModule`: Splits Agriculture-Vision dataset (80/20 train/val).
   - Batch size **16** with gradient accumulation **4** (effective batch = 64).
   - All CPU cores used for data loading with `prefetch_factor=4`.

### Hyperparameters

- **Learning Rate**: `6e-5` with **OneCycleLR** scheduler (10% warmup → cosine anneal).
- **Optimizer**: `AdamW` (weight decay 0.01).
- **Loss Function**: Combined `DiceLoss` + `BCEWithLogitsLoss`.
- **Precision**: `bf16-mixed` (BFloat16, native A100 format — no gradient scaling needed).
- **Epochs**: 7 (early stopping with patience 3).
- **Gradient Clipping**: 1.0 (prevents gradient explosion with high LR).

### A100 Optimizations

- **TF32 matmul**: `torch.set_float32_matmul_precision("high")` — ~3× faster than FP32 on A100.
- **cuDNN benchmark**: `torch.backends.cudnn.benchmark = True` — auto-tunes convolution kernels.
- **torch.compile**: Default mode — fuses ops and generates optimized Triton kernels.
- **BF16 mixed precision**: Native A100 format, avoids FP16 gradient scaling issues.

## Results

- **Validation IoU**: 0.729 (72.9%)
- **Validation Loss**: 0.173
- **Training Loss**: 0.151 (at epoch 7/7)

## Output

The final weights are saved to `models/weights/segformer_b4_farmtrack_final.pth`. These weights contain the state dict for the SegFormer-B4 model (MiT-B4 backbone).

Model size: ~240 MB

## How to Run

```bash
# On Google Colab with A100 runtime:
# 1. Mount Google Drive (for persistent checkpoint storage)
# 2. Install dependencies
pip install -r colab_a100_segformer/requirements.txt
# 3. Run training
python colab_a100_segformer/train_segformer_a100.py
```
