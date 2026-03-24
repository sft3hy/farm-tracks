# FarmTrack: Agricultural Equipment Track Detection

Detect and segment tracks left by farming equipment in agricultural fields using computer vision segmentation models.

## Project Structure
```
farmtrack/
├── data/
│   ├── raw/                  # Original satellite/drone images
│   ├── annotated/            # Labeled masks for training
│   └── processed/            # Preprocessed image tiles
├── models/
│   ├── weights/              # Saved model checkpoints
│   └── configs/              # Model configuration files
├── src/                      # Source code for data, models, training
├── notebooks/                # Exploratory data analysis & quality checks
└── app/                      # Gradio web UI demo
```

## Setup Instructions

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download data into `data/raw/`.
3. Preprocess data (tiling, augmentation) using scripts in `src/data/`.
4. Train models using `src/train.py`.

## Key Questions to Resolve
1. **Image source** — drone, satellite, or ground-level photos?
2. **Resolution** — tracks are easier to detect at <10cm/px (drone) vs. ~10m/px (Sentinel-2)
3. **Labeled data** — does he have existing annotations or are we starting cold?
4. **Output format** — does he need GeoJSON, a visual overlay, or database output?
5. **Real-time vs. batch** — periodic analysis pipeline or live drone feed?