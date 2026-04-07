#!/usr/bin/env python3
"""
FarmTrack Model Benchmark
=========================
Runs all three segmentation models (UNet, SegFormer, SAM) against the full
paired test set (RGB + planter_skip mask) and saves a permanent JSON report.

Usage:
    python src/run_benchmark.py

Memory: Loads one model at a time on MPS/CUDA/CPU. Batch size = 1.
        Safe for systems with 24 GB unified RAM.

Output: data/benchmark_report.json
"""

import os
import sys
import json
import glob
import time
import gc
import random
from datetime import datetime, timezone

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from datasets import Dataset, concatenate_datasets
from tqdm import tqdm

# ── path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.models.unet import UNetFarmTrack
from src.models.segformer import SegformerFarmTrack
from src.models.sam import SAMFarmTrack
from transformers import SamProcessor

# ── config ───────────────────────────────────────────────────────────────────
REPORT_PATH = os.path.join(ROOT, "data", "benchmark_report.json")
WEIGHTS_DIR = os.path.join(ROOT, "models", "weights")

# Device priority: MPS → CUDA → CPU
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

print(f"🖥  Device: {DEVICE}")

# ── transforms (must match training) ─────────────────────────────────────────
transform_rgb = T.Compose([
    T.ToTensor(),
    T.Resize((512, 512), antialias=True),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

transform_mask = T.Compose([
    T.ToTensor(),
    T.Resize((512, 512), antialias=True, interpolation=T.InterpolationMode.NEAREST),
])


# ── dataset loading ──────────────────────────────────────────────────────────
def load_paired_dataset():
    """Load Agriculture-Vision arrow files and pair RGB ↔ planter_skip masks across the whole fleet."""
    base_cache = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    ds_dir = os.path.join(base_cache, "datasets", "shi-labs___agriculture-vision")

    arrow_pattern = os.path.join(ds_dir, "**/agriculture-vision-train-*.arrow")
    arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))
    if not arrow_files:
        raise FileNotFoundError(f"No arrow files found in {ds_dir}")

    print(f"📦 Loading {len(arrow_files)} arrow shards...")
    # Only load keys to save memory and time during indexing
    ds = concatenate_datasets([Dataset.from_file(f).select_columns(['__key__']) for f in arrow_files])
    total = len(ds)
    print(f"   Total entries: {total:,}")

    # Exhaustive scan of all keys in the dataset
    rgb_map = {}
    mask_map = {}
    
    # We load all keys into memory for fast lookup (2M strings is ~200MB, well within 24GB)
    print(f"🔍 Indexing fleet for RGB/Mask pairs...")
    all_keys = ds["__key__"]
    
    for i, key in enumerate(tqdm(all_keys, desc="  Scanning keys")):
        # Use filename without extension as the base ID for matching
        file_id = os.path.splitext(key.split("/")[-1])[0]
        
        if "images/rgb" in key:
            rgb_map[file_id] = i
        elif "planter_skip" in key:
            mask_map[file_id] = i

    # Find common IDs present in both RGB and planter_skip
    common_ids = sorted(list(set(rgb_map.keys()) & set(mask_map.keys())))
    print(f"   Candidates: {len(rgb_map):,} RGB, {len(mask_map):,} Masks. Found {len(common_ids):,} potential pairs.")

    # Identify all valid fields (those with non-blank masks) across the entire fleet
    target_count = int(os.environ.get("BENCHMARK_SAMPLES", 1000))
    
    # Re-load full dataset (including images) for the exhaustive scan
    print("🔋 Exhaustively scanning ALL paired candidates for non-blank masks...")
    full_ds = concatenate_datasets([Dataset.from_file(f) for f in arrow_files])
    
    pairs = []
    # We scan all common_ids to find every single valid sample in the dataset
    for fid in tqdm(common_ids, desc="  Filtering fleet"):
        if len(pairs) >= target_count:
            break
            
        mask_sample = full_ds[mask_map[fid]]
        mask_img = mask_sample.get("png")
        if mask_img is None:
            continue
            
        # Fast check for non-zero pixels
        mask_arr = np.array(mask_img)
        if not np.any(mask_arr):
            continue
            
        # If valid, also fetch the RGB image
        rgb_sample = full_ds[rgb_map[fid]]
        rgb_img = rgb_sample.get("jpg") or rgb_sample.get("png")
        if rgb_img is None:
            continue
            
        pairs.append((fid, rgb_img, mask_img))

    print(f"✅ Found {len(pairs):,} non-blank fields across the entire Agriculture-Vision fleet.")
    return pairs


# ── metric computation ────────────────────────────────────────────────────────
def compute_metrics(pred_binary: np.ndarray, gt_binary: np.ndarray):
    """Compute IoU, F1, Precision, Recall between two binary masks."""
    intersection = float(np.sum(pred_binary * gt_binary))
    pred_sum = float(np.sum(pred_binary))
    gt_sum = float(np.sum(gt_binary))
    union = pred_sum + gt_sum - intersection

    iou = intersection / union if union > 0 else 0.0
    precision = intersection / pred_sum if pred_sum > 0 else 0.0
    recall = intersection / gt_sum if gt_sum > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "iou": round(iou, 5),
        "f1": round(f1, 5),
        "precision": round(precision, 5),
        "recall": round(recall, 5),
    }


# ── model loaders ─────────────────────────────────────────────────────────────
def load_model(name: str):
    """Load a single model onto DEVICE and return (model, is_sam)."""
    if name == "unet":
        m = UNetFarmTrack()
        wp = os.path.join(WEIGHTS_DIR, "unet_farmtrack_final.pth")
    elif name == "segformer":
        m = SegformerFarmTrack()
        wp = os.path.join(WEIGHTS_DIR, "segformer_farmtrack_final.pth")
    elif name == "segformer_b4":
        m = SegformerFarmTrack(pretrained_model_name="nvidia/mit-b4")
        wp = os.path.join(WEIGHTS_DIR, "segformer_b4_farmtrack_final.pth")
    elif name == "sam":
        m = SAMFarmTrack()
        wp = os.path.join(WEIGHTS_DIR, "sam_farmtrack_final.pth")
    else:
        raise ValueError(f"Unknown model: {name}")

    if os.path.exists(wp):
        m.load_state_dict(torch.load(wp, map_location="cpu"))
        print(f"   Loaded weights from {wp}")
    else:
        print(f"   ⚠ No weights found at {wp} — using random init")

    m = m.to(DEVICE)
    m.eval()
    return m


def unload_model(model):
    """Move model off device and free memory."""
    del model
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


# ── inference per model ───────────────────────────────────────────────────────
@torch.no_grad()
def evaluate_model(model_name: str, pairs: list):
    """Evaluate a single model on all pairs. Returns list of per-image dicts."""
    print(f"\n{'='*60}")
    print(f"  Evaluating: {model_name.upper()}")
    print(f"{'='*60}")

    model = load_model(model_name)
    is_sam = model_name == "sam"
    results = []

    for file_id, rgb_img, mask_img in tqdm(pairs, desc=f"  {model_name}"):
        # Prepare ground truth
        gt_tensor = transform_mask(mask_img.convert("L"))
        gt_binary = (gt_tensor > 0.5).float().squeeze().numpy()

        t0 = time.perf_counter()

        if is_sam:
            input_points = torch.tensor([[[256, 256]]], dtype=torch.float32).to(DEVICE)
            input_labels = torch.tensor([[1]], dtype=torch.long).to(DEVICE)
            logits = model(rgb_img.convert("RGB"), input_points=input_points, input_labels=input_labels)
        else:
            input_tensor = transform_rgb(rgb_img.convert("RGB")).unsqueeze(0).to(DEVICE)
            logits = model(input_tensor)

        latency_ms = (time.perf_counter() - t0) * 1000

        # Post-process prediction
        probs = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
        pred_binary = (probs > 0.5).astype(np.float32)

        # Resize pred to match GT if needed
        if pred_binary.shape != gt_binary.shape:
            from PIL import Image as PILImage
            pred_pil = PILImage.fromarray((pred_binary * 255).astype(np.uint8))
            pred_pil = pred_pil.resize((gt_binary.shape[1], gt_binary.shape[0]), PILImage.NEAREST)
            pred_binary = (np.array(pred_pil) / 255.0).astype(np.float32)

        metrics = compute_metrics(pred_binary, gt_binary)
        metrics["latency_ms"] = round(latency_ms, 2)
        metrics["file_id"] = file_id
        results.append(metrics)

    unload_model(model)
    return results


# ── aggregate stats ───────────────────────────────────────────────────────────
def aggregate(results: list):
    """Compute mean/median/std/min/max for each metric over all images."""
    if not results:
        return {}

    metrics_keys = ["iou", "f1", "precision", "recall", "latency_ms"]
    agg = {}
    for key in metrics_keys:
        vals = np.array([r[key] for r in results])
        agg[key] = {
            "mean": round(float(np.mean(vals)), 5),
            "median": round(float(np.median(vals)), 5),
            "std": round(float(np.std(vals)), 5),
            "min": round(float(np.min(vals)), 5),
            "max": round(float(np.max(vals)), 5),
        }
    return agg


def compute_histogram(results: list, key: str = "iou", bins: int = 20):
    """Compute histogram bin counts for a metric."""
    vals = [r[key] for r in results]
    counts, edges = np.histogram(vals, bins=bins, range=(0.0, 1.0))
    return {
        "counts": counts.tolist(),
        "bin_edges": [round(float(e), 4) for e in edges.tolist()],
    }


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  FarmTrack Model Benchmark")
    print("=" * 60)

    pairs = load_paired_dataset()
    if not pairs:
        print("❌ No paired fields found. Exiting.")
        sys.exit(1)

    model_names = ["unet", "segformer", "segformer_b4", "sam"]
    all_results = {}

    for mname in model_names:
        per_image = evaluate_model(mname, pairs)
        all_results[mname] = {
            "aggregate": aggregate(per_image),
            "histogram_iou": compute_histogram(per_image, "iou"),
            "histogram_f1": compute_histogram(per_image, "f1"),
            "per_image": per_image,
        }

    # ── winners ──
    winners = {}
    for metric in ["iou", "f1", "precision", "recall"]:
        best_model = max(model_names, key=lambda m: all_results[m]["aggregate"][metric]["mean"])
        winners[metric] = best_model
    # Latency: lower is better
    winners["latency_ms"] = min(model_names, key=lambda m: all_results[m]["aggregate"]["latency_ms"]["mean"])

    report = {
        "meta": {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "device": str(DEVICE),
            "total_samples": len(pairs),
            "dataset": "Agriculture-Vision (planter_skip)",
        },
        "winners": winners,
        "models": all_results,
    }

    # ── save ──
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n✅ Benchmark complete! Report saved to: {REPORT_PATH}")
    print(f"   Samples evaluated: {len(pairs)}")
    print()
    for mname in model_names:
        a = all_results[mname]["aggregate"]
        print(f"   {mname.upper():>12}  —  IoU: {a['iou']['mean']:.4f}  |  F1: {a['f1']['mean']:.4f}  |  Latency: {a['latency_ms']['mean']:.1f}ms")
    print(f"\n   🏆 Best IoU: {winners['iou'].upper()}")
    print(f"   🏆 Best F1:  {winners['f1'].upper()}")
    print(f"   ⚡ Fastest:  {winners['latency_ms'].upper()}")


if __name__ == "__main__":
    main()
