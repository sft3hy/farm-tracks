import pytorch_lightning as pl
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Image as HFImage, Value
from models.sam import SAMFarmTrack
import os
import numpy as np

class AgVisionSAMDataset(Dataset):
    def __init__(self, ds, index_list, mask_type="planter_skip"):
        self.ds = ds
        self.index_list = index_list
        self.mask_type = mask_type
        # SAM handles its own resize/normalization via processor, 
        # but we need to ensure the raw image is in a format SAM likes.

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_id, mapping = self.index_list[idx]
        
        # Load RGB
        rgb_idx = mapping.get("rgb")
        if rgb_idx is None:
            image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            sample = self.ds[rgb_idx]
            img = sample.get("jpg") or sample.get("png")
            image = np.array(img.convert("RGB"))

        # Load Mask
        mask_idx = mapping.get(self.mask_type)
        if mask_idx is None:
            mask = np.zeros((512, 512), dtype=np.float32)
        else:
            mask_sample = self.ds[mask_idx]
            mask_img = mask_sample.get("png")
            mask = np.array(mask_img.convert("L"))
            mask = (mask > 127).astype(np.float32)

        # Input points and labels for SAM
        # input_point: [[x, y]] -> shape (num_points, 2)
        # input_label: [1] -> shape (num_points,) -> 1 for foreground
        input_point = [[256, 256]] 
        input_label = [1]
        
        return {
            "image": image, # (H, W, 3)
            "mask": torch.from_numpy(mask).unsqueeze(0), # (1, H, W)
            "input_points": input_point,
            "input_labels": input_label
        }

class SAMFarmTrackModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        self.model = SAMFarmTrack()
        self.learning_rate = learning_rate
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        # Extract images, points and labels from batch
        # DataLoader will collate these into tensors/lists
        pixel_values = []
        for i in range(len(batch["image"])):
            img = batch["image"][i].cpu().numpy() if torch.is_tensor(batch["image"][i]) else batch["image"][i]
            pixel_values.append(img)
            
        points = batch["input_points"]
        labels = batch["input_labels"]
        
        # Forward pass through the model's adapter
        logits = self.model(pixel_values, input_points=points, input_labels=labels) 
        # logits shape: [B, 3, H, W] or [B, 1, H, W] depending on SAM output
        # SAM usually outputs 3 masks (multimask_output=True by default)
        
        # Just take the first mask for simplicity in this baseline
        if logits.dim() == 4 and logits.shape[1] > 1:
            logits = logits[:, 0:1, :, :]
            
        # Target mask resize to match logits if necessary
        y = batch["mask"]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear")

        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        # Only optimize the mask decoder (as specified in sam.py adapter)
        # However, SAMFarmTrack freezes vision/prompt encoders in __init__.
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    try:
        from train import FarmTrackDataModule
    except ImportError:
        from src.train import FarmTrackDataModule
    
    # We need to ensure we use the specialized SAM dataset
    class SAMDataModule(FarmTrackDataModule):
        def train_dataloader(self):
            return DataLoader(AgVisionSAMDataset(self.ds, self.train_idx, self.mask_type), 
                              batch_size=2, shuffle=True, num_workers=2) # SAM is heavy
        def val_dataloader(self):
            return DataLoader(AgVisionSAMDataset(self.ds, self.val_idx, self.mask_type), 
                              batch_size=2, shuffle=False, num_workers=2)

    os.makedirs("models/weights", exist_ok=True)
    module = SAMFarmTrackModule()
    datamodule = SAMDataModule(batch_size=2)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/weights/",
        filename="sam-farmtrack-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        max_epochs=5, # SAM fine-tuning is faster to converge
        callbacks=[checkpoint_callback],
        accelerator="cuda" if torch.cuda.is_available() else "auto",
        devices=1,
        precision="16-mixed", # T4 optimization
    )
    
    print("🚀 Starting CUDA SAM Training (Fine-tuning Mask Decoder)...")
    trainer.fit(module, datamodule=datamodule)
    
    final_path = "models/weights/sam_farmtrack_final.pth"
    torch.save(module.model.state_dict(), final_path)
    print(f"✅ Training Complete! Saved to {final_path}")
