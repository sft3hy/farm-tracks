import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np

class FarmTrackDataset(Dataset):
    """
    Dataset class for loading agricultural track images and masks.
    """
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        
        self.images = [f for f in sorted(os.listdir(image_dir)) if f.endswith(('.png', '.jpg', '.tif'))]
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name)
        
        # Load image (BGR -> RGB)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask (Grayscale)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # Binary mask assuming tracks are 255 and background is 0
        mask = (mask > 127).astype(np.float32)
        
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # Add channel dimension to mask
        mask = np.expand_dims(mask, axis=0)
            
        return {"image": image, "mask": mask}
