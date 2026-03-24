import torch
import cv2
import numpy as np
from torchvision import transforms

class FarmTrackInferencer:
    def __init__(self, model, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.model = model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def predict(self, image_path, threshold=0.5):
        """
        Runs inference on a single image and returns the binary mask.
        """
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")
            
        original_size = image.shape[:2]
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Typically you'd resize or tile the image here
        # For simplicity, resize to 512x512
        resized = cv2.resize(image, (512, 512))
        
        tensor = self.transform(resized).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.sigmoid(logits).squeeze().cpu().numpy()
            
        mask = (probs > threshold).astype(np.uint8) * 255
        
        # Resize mask back to original
        mask = cv2.resize(mask, (original_size[1], original_size[0]), interpolation=cv2.INTER_NEAREST)
        return mask
