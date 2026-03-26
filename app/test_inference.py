import os
import sys
import torch
import numpy as np
import torchvision.transforms as T
import matplotlib.pyplot as plt
from PIL import Image
from datasets import load_dataset, Features, Image as HFImage, Value

# Ensure we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.models.unet import UNetFarmTrack


def main():
    # 1. Setup Transforms exactly as in train.py
    transform_rgb = T.Compose(
        [
            T.ToTensor(),
            T.Resize((512, 512), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    transform_mask = T.Compose(
        [
            T.ToTensor(),
            T.Resize(
                (512, 512), antialias=True, interpolation=T.InterpolationMode.NEAREST
            ),
        ]
    )

    # 2. Load Dataset Structure
    print("Loading Hugging Face dataset...")
    features = Features(
        {
            "png": HFImage(),
            "jpg": HFImage(),
            "__key__": Value("string"),
            "__url__": Value("string"),
        }
    )
    # Using the same split the model was trained/validated on
    ds = load_dataset("shi-labs/Agriculture-Vision", split="train", features=features)

    # 3. Build the index mapping (matching train.py exactly)
    print("Building index mapping for dataset pairs...")
    mapping = {}
    for i, sample in enumerate(ds):
        key = sample.get("__key__", "")
        if not key:
            continue

        parts = key.split("/")
        file_id = parts[-1]
        folder = parts[-2] if len(parts) >= 2 else None

        if file_id not in mapping:
            mapping[file_id] = {}

        if folder == "rgb":
            mapping[file_id]["rgb"] = i
        elif folder == "planter_skip":
            mapping[file_id]["mask"] = i

    # Filter out anything that doesn't have both an RGB image and a planter_skip mask
    index_list = [(k, v) for k, v in mapping.items() if "rgb" in v and "mask" in v]
    print(f"Found {len(index_list)} complete image/mask pairs.")

    # 4. Find 5 pairs that actually contain Ground Truth anomalies
    print("Searching for 5 fields that have actual 'planter_skip' anomalies to test...")
    test_pairs = []
    for file_id, idx_map in index_list:
        mask_idx = idx_map["mask"]
        mask_sample = ds[mask_idx]
        mask_img = mask_sample.get("png")

        # Quick check: does the mask have positive pixels?
        mask_tensor = transform_mask(mask_img.convert("L"))
        gt_binary = (mask_tensor > 0.5).float()

        if (
            gt_binary.sum().item() > 100
        ):  # Ensures there is a visible anomaly (>100 pixels)
            test_pairs.append((file_id, idx_map, gt_binary))

        if len(test_pairs) >= 5:
            break

    # 5. Load Model
    print("\nLoading UNetFarmTrack Model...")
    model = UNetFarmTrack()
    weights_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..",
            "models",
            "weights",
            "unet_farmtrack_final.pth",
        )
    )

    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
        print(f"Model weights loaded from {weights_path}.")
    else:
        print(
            f"WARNING: Weights not found at {weights_path}. Testing with random initialization!"
        )
    model.eval()

    # 6. Run Inference & Plot Outputs
    output_dir = os.path.join(os.path.dirname(__file__), "..", "test_outputs")
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nEvaluating on {len(test_pairs)} images...")

    for i, (file_id, idx_map, gt_mask_tensor) in enumerate(test_pairs):
        print(f"\n--- Processing Image {i+1}: {file_id} ---")

        # Load the corresponding RGB image
        rgb_idx = idx_map["rgb"]
        rgb_sample = ds[rgb_idx]
        rgb_img = rgb_sample.get("jpg") or rgb_sample.get("png")

        # Prepare Input
        input_tensor = transform_rgb(rgb_img.convert("RGB")).unsqueeze(0)

        # Inference
        with torch.no_grad():
            logits = model(input_tensor)
            probs = torch.sigmoid(logits)

            probs_np = probs.squeeze().cpu().numpy()
            pred_mask = (probs_np > 0.5).astype(np.uint8)
            gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy()

            print(
                f"Logits min/max/mean: {logits.min().item():.2f} / {logits.max().item():.2f} / {logits.mean().item():.2f}"
            )
            print(
                f"Probs min/max/mean:  {probs.min().item():.5f} / {probs.max().item():.5f} / {probs.mean().item():.5f}"
            )
            print(f"Ground Truth Anomalous Pixels: {np.sum(gt_mask_np)}")
            print(f"Predicted Anomalous Pixels (>0.5): {np.sum(pred_mask)}")

            # --- Visualization ---
            plt.figure(figsize=(15, 5))

            # Subplot 1: Input Image (Denormalized)
            plt.subplot(1, 3, 1)
            plt.title("Input RGB Image")
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            vis_img = input_tensor.squeeze().permute(1, 2, 0).numpy()
            vis_img = std * vis_img + mean
            vis_img = np.clip(vis_img, 0, 1)  # Keep in 0-1 range for matplotlib
            plt.imshow(vis_img)
            plt.axis("off")

            # Subplot 2: Ground Truth Label
            plt.subplot(1, 3, 2)
            plt.title(f"Ground Truth Mask\n({int(np.sum(gt_mask_np))} pixels)")
            plt.imshow(gt_mask_np, cmap="gray")
            plt.axis("off")

            # Subplot 3: Prediction
            plt.subplot(1, 3, 3)
            plt.title(f"Predicted Mask\n({np.sum(pred_mask)} pixels)")
            plt.imshow(pred_mask, cmap="gray")
            plt.axis("off")

            # Save the figure
            save_path = os.path.join(
                output_dir, f"inference_{file_id.split('.')[0]}.png"
            )
            plt.savefig(save_path, bbox_inches="tight")
            plt.close()
            print(f"Saved visualization to {save_path}")


if __name__ == "__main__":
    main()
