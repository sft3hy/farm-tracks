import gradio as gr
import cv2
import numpy as np
from PIL import Image
from datasets import load_dataset, Features, Image as HFImage, Value
import logging
import sys
import torch
import torchvision.transforms as T
import os
import sys

# Ensure src module is reachable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

DS = None

def get_dataset():
    global DS
    if DS is None:
        try:
            logger.info("Initializing Agriculture-Vision dataset with custom schema...")
            features = Features({
                'png': HFImage(),
                'jpg': HFImage(),
                '__key__': Value('string'),
                '__url__': Value('string')
            })
            DS = load_dataset(
                "shi-labs/Agriculture-Vision", 
                split="train", 
                features=features
            )
            logger.info(f"Dataset loaded successfully with {len(DS)} total examples.")
        except Exception as e:
            logger.error(f"Error loading Agriculture-Vision dataset: {e}")
            return None
    return DS

def load_batch(state):
    logger.info("Request received to load the next batch of images.")
    ds = get_dataset()
    if ds is None:
        logger.warning("Dataset is not loaded. Returning empty batch.")
        return state, []
    
    start_idx = state.get('index', 0)
    logger.info(f"Starting to pull images from dataset index {start_idx}.")
    batch_images = []
    
    idx = start_idx
    # Scan forward to find exactly 10 RGB images
    while len(batch_images) < 10 and idx < len(ds):
        sample = ds[idx]
        key = sample.get('__key__', '')
        
        # The agriculture-vision dataset flattens all masks and images. We only want RGB inputs.
        if '/images/rgb/' in key:
            img = sample.get('jpg') or sample.get('png')
            if img is not None:
                filename = key.split('/')[-1]
                batch_images.append((img, filename))
            
        idx += 1
        
    # Wrap around if we hit the end
    state['index'] = idx % len(ds)
    logger.info(f"Successfully loaded {len(batch_images)} RGB images. New start index is {state['index']}.")
    return state, batch_images

try:
    from src.models.unet import UNetFarmTrack
    segmentation_model = UNetFarmTrack()
    
    # Load compiled weights if the user has completed training!
    weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models', 'weights', 'unet_farmtrack_final.pth'))
    if os.path.exists(weights_path):
        segmentation_model.load_state_dict(torch.load(weights_path, map_location='cpu'))
        logger.info(f"SUCCESS: Loaded trained UNetFarmTrack weights from {weights_path}.")
    else:
        logger.warning("UNetFarmTrack initialized with random weights! Run `python src/train.py` to compile trained weights.")
        
    # If using mac silicon or CUDA
    if torch.backends.mps.is_available():
        segmentation_model = segmentation_model.to('mps')
    elif torch.cuda.is_available():
        segmentation_model = segmentation_model.to('cuda')
    segmentation_model.eval()
    logger.info("UNetFarmTrack model loaded successfully (untrained weights).")
except Exception as e:
    logger.error(f"Failed to load UNetFarmTrack: {e}")
    segmentation_model = None

# Real prediction function using UNet
@torch.no_grad()
def predict_tracks(image_np):
    """
    Takes a numpy image, runs UNetFarmTrack inference, and returns an image with the overlay.
    """
    if image_np is None:
        return None
        
    if segmentation_model is None:
        return image_np
        
    device = next(segmentation_model.parameters()).device
    
    # Preprocess image for ResNet backbone (ImageNet normalization)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((512, 512), antialias=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Convert RGB numpy to PyTorch tensor [1, C, H, W]
    input_tensor = transform(image_np).unsqueeze(0).to(device)
    
    # Inference
    logits = segmentation_model(input_tensor) # [1, 1, H, W]
    probs = torch.sigmoid(logits).squeeze().cpu().numpy()
    
    # Binarize prediction
    binary_mask = (probs > 0.5).astype(np.uint8)
    
    # Resize mask back to original image size
    h, w = image_np.shape[:2]
    binary_mask = cv2.resize(binary_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create red mask for tracks
    color_mask = np.zeros_like(image_np)
    color_mask[binary_mask > 0] = [255, 0, 0] # Red tracks
    
    # Blend overlay
    blended = cv2.addWeighted(image_np, 0.7, color_mask, 0.3, 0)
    return blended

def predict_batch(gallery_images):
    if not gallery_images:
        logger.warning("No images provided in the gallery for track detection.")
        return []
    
    logger.info(f"Starting track detection for a batch of {len(gallery_images)} images.")
    output_images = []
    for i, item in enumerate(gallery_images):
        # Handle different Gradio Gallery output formats across 3.x and 4.x
        if hasattr(item, 'items') or isinstance(item, dict):
            # Gradio 3.x format: dict with 'name' being the filepath
            path = item.get('name') or item.get('image')
            caption = item.get('info', f"Image {i}")
        elif isinstance(item, tuple) or isinstance(item, list):
            # Gradio 4.x format: tuple of (filepath, caption)
            path = item[0]
            caption = item[1] if len(item) > 1 else f"Image {i}"
        else:
            # Fallback
            path = str(item)
            caption = f"Image {i}"
            
        # Load the image from the path returned by Gradio
        img = Image.open(path).convert('RGB')
        img_np = np.array(img)
        
        blended = predict_tracks(img_np)
        output_images.append((blended, caption))
        
    logger.info("Track detection complete for the batch.")
    return output_images

def build_app():
    with gr.Blocks(title="FarmTrack Demo") as demo:
        gr.Markdown("# 🌾 FarmTrack: Equipment Track Detection")
        gr.Markdown("Click 'Load Next Batch' to fetch 10 fields from the Agriculture-Vision dataset, then click 'Detect Tracks' to segment equipment tracks.")
        
        dataset_state = gr.State(value={'index': 0})
        
        with gr.Row():
            load_btn = gr.Button("Load Next Batch (10 Images)")
        
        with gr.Row():
            with gr.Column():
                input_gallery = gr.Gallery(label="Input Field Images (Batch of 10)", columns=5, rows=2, object_fit="contain", height="auto")
                run_btn = gr.Button("Detect Tracks", variant="primary")
            
            with gr.Column():
                output_gallery = gr.Gallery(label="Detected Tracks Overlay (Batch of 10)", columns=5, rows=2, object_fit="contain", height="auto")
                
        load_btn.click(load_batch, inputs=dataset_state, outputs=[dataset_state, input_gallery])
        run_btn.click(predict_batch, inputs=input_gallery, outputs=output_gallery)
        
    return demo

if __name__ == "__main__":
    logger.info("Starting FarmTrack Gradio application...")
    try:
        # Initialize dataset so it's ready when the server starts
        get_dataset()
        app = build_app()
        logger.info("Launching Gradio server...")
        app.launch()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received.")
    finally:
        logger.info("Application shutting down.")
