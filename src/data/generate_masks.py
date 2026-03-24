import os
import cv2
import numpy as np
import glob

def generate_simulated_masks(raw_dir="data/raw", annotated_dir="data/annotated"):
    os.makedirs(annotated_dir, exist_ok=True)
    images = glob.glob(os.path.join(raw_dir, "*.jpg"))
    print(f"Generating simulated masks for {len(images)} images...")
    
    for img_path in images:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
            
        # Use a combination of blur and canny to find heavy lines (tracks)
        blurred = cv2.GaussianBlur(img, (9, 9), 0)
        edges = cv2.Canny(blurred, 30, 100)
        
        # Dilate edges to make thicker masks
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Save exact same basename
        basename = os.path.basename(img_path)
        out_path = os.path.join(annotated_dir, basename.replace(".jpg", ".png"))
        cv2.imwrite(out_path, dilated)
        print(f"Saved mask {out_path}")

if __name__ == "__main__":
    generate_simulated_masks()
