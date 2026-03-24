from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import glob
import cv2
import base64
import numpy as np
import io
from PIL import Image

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

RAW_DIR = "data/raw"
os.makedirs(RAW_DIR, exist_ok=True)

@app.get("/images")
def list_images():
    """List all available raw images."""
    paths = glob.glob(os.path.join(RAW_DIR, "*.jpg")) + glob.glob(os.path.join(RAW_DIR, "*.png"))
    basenames = [os.path.basename(p) for p in paths]
    return {"images": basenames}

@app.get("/image/{filename}")
def get_image(filename: str):
    """Serve raw image bytes."""
    path = os.path.join(RAW_DIR, filename)
    if os.path.exists(path):
        return FileResponse(path)
    return JSONResponse(status_code=404, content={"message": "Image not found"})

@app.post("/infer/{filename}")
async def infer_image(filename: str):
    """
    Run inference on a given filename in the raw directory.
    Returns the predicted mask as base64 png, and dummy evaluation metrics.
    """
    path = os.path.join(RAW_DIR, filename)
    if not os.path.exists(path):
        return JSONResponse(status_code=404, content={"message": "Image not found"})
        
    img = cv2.imread(path)
    
    # ----------------------------------------------------
    # MOCK INFERENCE (until UNet is fully trained)
    # Using edge detection to mock the "predicted mask"
    # ----------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(edges, kernel, iterations=1)
    
    # Encode mask to base64
    _, buffer = cv2.imencode('.png', mask)
    mask_b64 = base64.b64encode(buffer).decode('utf-8')
    
    # Calculate mock metrics (simulating IoU against the generated ground truth)
    metrics = {
        "mIoU": round(np.random.uniform(0.65, 0.85), 3),
        "f1Score": round(np.random.uniform(0.70, 0.90), 3)
    }
    
    return {
        "mask_base64": mask_b64,
        "metrics": metrics
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
