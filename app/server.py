from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
import os
import cv2
import base64
import numpy as np
import io
import torch
import torchvision.transforms as T
from PIL import Image
from datasets import Dataset, concatenate_datasets
import logging
import sys
import threading
import glob

# Ensure src module is reachable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.models.unet import UNetFarmTrack
from src.models.segformer import SegformerFarmTrack
from src.models.sam import SAMFarmTrack
from app.reporting import PerformanceReportManager

# Global state
report_manager = PerformanceReportManager(data_dir="data")
dataset_ready_event = threading.Event()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Global State ---
INDEX_LIST = None  # List of (file_id, {rgb_img: PIL.Image, mask_img: PIL.Image})
DS = None  # Full dataset for random access
loaded_models = {}  # Cache for loaded model instances
current_page = 0
BATCH_SIZE = 10
loading_status = {"state": "idle", "progress": 0, "total": 0, "found": 0}
loading_lock = threading.Lock()

# --- Transforms (matching train.py exactly) ---
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
        T.Resize((512, 512), antialias=True, interpolation=T.InterpolationMode.NEAREST),
    ]
)


def _load_dataset_background():
    """Background thread: loads dataset and builds index mapping with cached images."""
    global DS, INDEX_LIST, loading_status

    with loading_lock:
        loading_status["state"] = "loading_dataset"

    try:
        # 1. Try to find the dataset path from environment or use a sensible default
        base_cache = os.environ.get(
            "HF_HOME", os.path.expanduser("~/.cache/huggingface")
        )
        ds_dir = os.path.join(base_cache, "datasets", "shi-labs___agriculture-vision")

        logger.info(
            f"Background: Searching for Agriculture-Vision arrow files in {ds_dir}..."
        )

        # We use a recursive glob to find the arrow files regardless of the exact hash subfolder
        arrow_pattern = os.path.join(ds_dir, "**/agriculture-vision-train-*.arrow")
        arrow_files = sorted(glob.glob(arrow_pattern, recursive=True))

        if not arrow_files:
            # Fallback for older or different cache structures
            logger.warning(
                "Could not find arrow files with recursive glob. Checking legacy paths."
            )
            legacy_path = "/root/.cache/huggingface/datasets/shi-labs___agriculture-vision/default-5eb8c5b2696fef73/0.0.0/3d7d6c5fbf08e3d05aff6d04d0ea9dcb846b4acb/agriculture-vision-train-*.arrow"
            arrow_files = sorted(glob.glob(legacy_path))

        if not arrow_files:
            raise FileNotFoundError(
                f"Could not find local arrow files in {ds_dir} or fallback paths!"
            )

        # logger.info(f"Background: Found {len(arrow_files)} arrow files: {arrow_files}")

        DS = concatenate_datasets([Dataset.from_file(f) for f in arrow_files])
        total = len(DS)
        logger.info(f"Background: Dataset loaded instantly. {total} total entries.")

        with loading_lock:
            loading_status["state"] = "building_index"
            loading_status["total"] = total

        # --- Targeted scanning ---
        # The dataset is sorted by folder path. Known ranges:
        #   train/images/rgb      starts at ~431,062
        #   train/labels/planter_skip starts at ~680,000
        # We scan each range separately and pair by file_id.
        MAX_PER_RANGE = int(os.environ.get("MAX_DATASET_SCAN", 10000))

        # 1. Scan RGB entries
        logger.info(
            f"Background: Scanning RGB entries (starting at ~431062, max {MAX_PER_RANGE})..."
        )
        rgb_map = {}
        RGB_START = 431062
        for i in range(RGB_START, min(RGB_START + MAX_PER_RANGE, total)):
            key = DS[i]["__key__"]
            if "images/rgb" not in key:
                if i > RGB_START + 100:  # past the RGB block
                    break
                continue
            file_id = key.split("/")[-1]
            rgb_map[file_id] = i

            if (i - RGB_START) % 500 == 0:
                with loading_lock:
                    loading_status["progress"] = i - RGB_START
                    loading_status["found"] = len(rgb_map)
                logger.info(
                    f"  RGB: scanned {i - RGB_START}, found {len(rgb_map)} images..."
                )

        logger.info(f"Background: Found {len(rgb_map)} RGB images.")

        # 2. Scan planter_skip entries
        logger.info(
            f"Background: Scanning planter_skip entries (starting at ~680000, max {MAX_PER_RANGE})..."
        )
        mask_map = {}
        MASK_START = 680000
        for i in range(MASK_START, min(MASK_START + MAX_PER_RANGE, total)):
            key = DS[i]["__key__"]
            if "planter_skip" not in key:
                if i > MASK_START + 100:
                    break
                continue
            file_id = key.split("/")[-1]
            mask_map[file_id] = i

            if (i - MASK_START) % 500 == 0:
                logger.info(
                    f"  Masks: scanned {i - MASK_START}, found {len(mask_map)} masks..."
                )

        logger.info(f"Background: Found {len(mask_map)} planter_skip masks.")

        # 3. Pair by file_id and cache images
        logger.info("Background: Pairing RGB images with masks and caching...")
        paired_ids = set(rgb_map.keys()) & set(mask_map.keys())
        mapping = {}
        skipped = 0
        for j, file_id in enumerate(paired_ids):
            rgb_sample = DS[rgb_map[file_id]]
            mask_sample = DS[mask_map[file_id]]
            rgb_img = rgb_sample.get("jpg") or rgb_sample.get("png")
            mask_img = mask_sample.get("png")

            if rgb_img is None or mask_img is None:
                skipped += 1
                continue

            # Filter out images that have no anomalies to ensure the user sees interesting things
            if not np.any(np.array(mask_img)):
                skipped += 1
                continue

            mapping[file_id] = {
                "rgb_img": rgb_img,
                "mask_img": mask_img,
            }
            if j % 100 == 0:
                with loading_lock:
                    loading_status["found"] = len(mapping)
                logger.info(
                    f"  Cached {len(mapping)} valid / {j} scanned ({skipped} skipped)..."
                )

        logger.info(f"Background: {skipped} pairs skipped due to missing image data.")
        INDEX_LIST = list(mapping.items())

        with loading_lock:
            loading_status["state"] = "ready"
            loading_status["found"] = len(INDEX_LIST)
            loading_status["progress"] = total

        logger.info(f"Background: DONE. Found {len(INDEX_LIST)} paired fields.")
        dataset_ready_event.set()
        
        # Trigger report generation if not already present
        if report_manager.get_results() is None:
            logger.info("Automatically starting performance report generation...")
            # We wrap infer_image in a synchronous-looking wrapper or use asyncio
            # In reporting.py we used asyncio.run(infer_image_func)
            report_manager.start_report_generation(INDEX_LIST, get_model, infer_image)
    except Exception as e:
        logger.error(f"Background loading failed: {e}", exc_info=True)
        with loading_lock:
            loading_status["state"] = "error"


def get_model(name: str = "unet"):
    global loaded_models
    name = name.lower()

    if name in loaded_models:
        return loaded_models[name]

    try:
        if name == "unet":
            m = UNetFarmTrack()
            weights_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "models",
                    "weights",
                    "unet_farmtrack_final.pth",
                )
            )
            model_label = "UNetFarmTrack"
        elif name == "segformer":
            m = SegformerFarmTrack()
            weights_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "models",
                    "weights",
                    "segformer_farmtrack_final.pth",
                )
            )
            model_label = "SegformerFarmTrack"
        elif name == "sam":
            m = SAMFarmTrack()
            weights_path = os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__),
                    "..",
                    "models",
                    "weights",
                    "sam_farmtrack_final.pth",
                )
            )
            model_label = "SAMFarmTrack"
        else:
            raise ValueError(f"Unknown model name: {name}")

        if os.path.exists(weights_path):
            m.load_state_dict(torch.load(weights_path, map_location="cpu"))
            logger.info(
                f"SUCCESS: Loaded trained {model_label} weights from {weights_path}."
            )
        else:
            logger.warning(
                f"{model_label} initialized with random weights! Expected at {weights_path}"
            )

        if torch.backends.mps.is_available():
            m = m.to("mps")
        elif torch.cuda.is_available():
            m = m.to("cuda")

        m.eval()
        loaded_models[name] = m
        return m
    except Exception as e:
        logger.error(f"Failed to load model {name}: {e}", exc_info=True)
        return None


@app.on_event("startup")
async def startup_event():
    """Start dataset loading in a background thread so the API is immediately responsive."""
    thread = threading.Thread(target=_load_dataset_background, daemon=True)
    thread.start()
    logger.info("Dataset loading started in background thread. API is available now.")


@app.get("/status")
async def get_status():
    """Return the current loading status so the frontend can show progress."""
    with loading_lock:
        return loading_status.copy()


@app.get("/batch")
async def get_batch(model: str = "unet"):
    """Return the next page of 10 paired fields with thumbnails and model-specific inference."""
    global current_page

    if INDEX_LIST is None or len(INDEX_LIST) == 0:
        with loading_lock:
            return {
                "batch": [],
                "page": 0,
                "total_fields": 0,
                "loading": loading_status.copy(),
            }

    start = current_page * BATCH_SIZE
    end = start + BATCH_SIZE

    if start >= len(INDEX_LIST):
        current_page = 0
        start = 0
        end = BATCH_SIZE

    page_items = INDEX_LIST[start:end]
    current_page += 1

    batch = []
    for file_id, cache in page_items:
        img = cache["rgb_img"]

        thumb = img.copy()
        thumb.thumbnail((256, 256))
        buffered = io.BytesIO()
        thumb.save(buffered, format="JPEG")
        thumb_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

        batch.append(
            {
                "file_id": file_id,
                "thumbnail": thumb_b64,
                "inference": await infer_image(file_id, model=model),
            }
        )

    logger.info(f"Served batch page {current_page - 1} ({len(batch)} items)")
    return {"batch": batch, "page": current_page - 1, "total_fields": len(INDEX_LIST)}


@app.get("/image/{file_id}")
async def get_image(file_id: str):
    """Return the full-resolution RGB image for this file_id."""
    try:
        if INDEX_LIST is None:
            raise HTTPException(status_code=503, detail="Dataset still loading")

        entry = next((v for k, v in INDEX_LIST if k == file_id), None)
        if entry is None or "rgb_img" not in entry:
            raise HTTPException(status_code=404, detail=f"Image not found: {file_id}")

        img = entry["rgb_img"]
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format="JPEG")
        return Response(content=img_byte_arr.getvalue(), media_type="image/jpeg")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in get_image for {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@torch.no_grad()
@app.post("/infer/{file_id}")
async def infer_image(file_id: str, model: str = "unet"):
    """Run inference on the RGB image using selected model and compare against GT mask."""
    try:
        if INDEX_LIST is None:
            raise HTTPException(status_code=503, detail="Dataset still loading")

        segmentation_model = get_model(model)

        entry = next((v for k, v in INDEX_LIST if k == file_id), None)
        if entry is None or "rgb_img" not in entry:
            raise HTTPException(status_code=404, detail=f"Image not found: {file_id}")

        img = entry["rgb_img"]
        input_tensor = transform_rgb(img.convert("RGB")).unsqueeze(0)

        if segmentation_model is None:
            logger.warning("No model available, returning empty mask")
            mask = np.zeros((512, 512), dtype=np.uint8)
        elif model == "sam":
            # SAM expects raw-ish images (PIL or numpy) and points.
            # We pass the PIL image directly and a central prompt point.
            device = next(segmentation_model.parameters()).device
            input_points = torch.tensor([[[256, 256]]], dtype=torch.float32).to(device)
            input_labels = torch.tensor([[1]], dtype=torch.long).to(device)
            
            logits = segmentation_model(img.convert("RGB"), input_points=input_points, input_labels=input_labels)
            probs = torch.sigmoid(logits).detach().cpu().squeeze().numpy()
            
            binary_mask = (probs > 0.5).astype(np.uint8)
            mask = (binary_mask * 255).astype(np.uint8)
        else:
            device = next(segmentation_model.parameters()).device
            input_tensor = input_tensor.to(device)
            logits = segmentation_model(input_tensor)
            probs = torch.sigmoid(logits).detach().cpu().squeeze().numpy()

            binary_mask = (probs > 0.5).astype(np.uint8)
            mask = (binary_mask * 255).astype(np.uint8)

            logger.info(
                f"Inference on {file_id}: logits [{logits.min().item():.2f}, {logits.max().item():.2f}], probs max {probs.max():.4f}, positive px: {np.sum(binary_mask)}"
            )

        # Create BGRA image: Hot Pink (B:180, G:105, R:255, A:255)
        colored_pred = np.zeros((mask.shape[0], mask.shape[1], 4), dtype=np.uint8)
        colored_pred[mask == 255] = [180, 105, 255, 255]
        _, buffer = cv2.imencode(".png", colored_pred)
        pred_mask_b64 = base64.b64encode(buffer).decode("utf-8")

        metrics = {"mIoU": 0.0, "f1Score": 0.0}
        gt_mask_b64 = None
        if "mask_img" in entry:
            try:
                gt_tensor = transform_mask(entry["mask_img"].convert("L"))
                gt_binary = (gt_tensor > 0.5).float().squeeze().numpy()

                pred_binary = (mask / 255.0).astype(np.float32)
                if pred_binary.shape != gt_binary.shape:
                    pred_binary = cv2.resize(
                        pred_binary,
                        (gt_binary.shape[1], gt_binary.shape[0]),
                        interpolation=cv2.INTER_NEAREST,
                    )

                intersection = np.sum(pred_binary * gt_binary)
                union = np.sum(pred_binary) + np.sum(gt_binary) - intersection
                iou = float(intersection / union) if union > 0 else 0.0

                precision = (
                    float(intersection / np.sum(pred_binary))
                    if np.sum(pred_binary) > 0
                    else 0.0
                )
                recall = (
                    float(intersection / np.sum(gt_binary))
                    if np.sum(gt_binary) > 0
                    else 0.0
                )
                f1 = (
                    float(2 * precision * recall / (precision + recall))
                    if (precision + recall) > 0
                    else 0.0
                )

                metrics = {
                    "mIoU": round(iou, 4), 
                    "f1Score": round(f1, 4),
                    "precision": round(precision, 4),
                    "recall": round(recall, 4)
                }
                logger.info(f"Metrics for {file_id}: IoU={iou:.4f}, F1={f1:.4f}, P={precision:.4f}, R={recall:.4f}")

                gt_mask_uint8 = (gt_binary * 255).astype(np.uint8)
                colored_gt = np.zeros(
                    (gt_mask_uint8.shape[0], gt_mask_uint8.shape[1], 4), dtype=np.uint8
                )
                colored_gt[gt_mask_uint8 == 255] = [180, 105, 255, 255]
                _, gt_buffer = cv2.imencode(".png", colored_gt)
                gt_mask_b64 = base64.b64encode(gt_buffer).decode("utf-8")
            except Exception as e:
                logger.warning(f"Could not process GT mask for {file_id}: {e}")

        return {
            "mask_base64": pred_mask_b64,
            "gt_mask_base64": gt_mask_b64,
            "metrics": metrics,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in infer_image for {file_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/report/status")
async def get_report_status():
    return report_manager.get_status()

@app.get("/report/results")
async def get_report_results():
    results = report_manager.get_results()
    if results is None:
        raise HTTPException(status_code=404, detail="Report results not ready or found")
    return results

@app.get("/training-explanation/{model}")
async def get_training_explanation(model: str):
    """Serve the training documentation for a specific model."""
    try:
        model = model.lower()
        if model not in ["unet", "segformer", "sam"]:
            raise HTTPException(status_code=404, detail="Model documentation not found")
        
        file_path = os.path.abspath(
            os.path.join(
                os.path.dirname(__file__),
                "..",
                "src",
                "models",
                "training-explanation",
                f"train_{model}.md"
            )
        )
        
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail=f"Documentation file not found: {file_path}")
            
        with open(file_path, "r") as f:
            content = f.read()
        return {"content": content}
    except Exception as e:
        logger.error(f"Error serving documentation for {model}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/compare")
async def compare_models_legacy():
    # Keep this for legacy compatibility with frontend or redirect
    results = report_manager.get_results()
    if results:
        # Format results for legacy compare endpoint
        return {
            "comparison": results["metrics"],
            "winner_iou": results["summary"]["winners"].get("mIoU"),
            "winner_f1": results["summary"]["winners"].get("mF1"),
            "sample_count": results["summary"]["sample_count"]
        }
    else:
        # If not ready, return current status or error
        status = report_manager.get_status()
        return {"state": status["state"], "progress": status["progress"], "total": status["total"]}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
