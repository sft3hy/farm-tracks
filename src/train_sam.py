import pytorch_lightning as pl
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from models.sam import SAMFarmTrack
from transformers import SamProcessor
import os
import numpy as np

# Optimize CUDA memory allocation to prevent fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from data.ag_vision import FarmTrackDataModule, AgVisionSAMDataset

class SAMFarmTrackModule(pl.LightningModule):
    def __init__(self, learning_rate=1e-5):
        super().__init__()
        self.model = SAMFarmTrack()
        self.learning_rate = learning_rate
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def training_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        points = batch["input_points"]
        labels = batch["input_labels"]
        
        # Forward pass through the model's adapter
        logits = self.model(pixel_values, input_points=points, input_labels=labels) 
            
        # Target mask resize to match logits if necessary
        y = batch["mask"]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear")

        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values = batch["pixel_values"]
        points = batch["input_points"]
        labels = batch["input_labels"]
        
        logits = self.model(pixel_values, input_points=points, input_labels=labels)
        
        y = batch["mask"]
        if logits.shape[-2:] != y.shape[-2:]:
            logits = nn.functional.interpolate(logits, size=y.shape[-2:], mode="bilinear")
            
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

if __name__ == "__main__":
    import sys
    # Add project root to sys.path if needed
    # (Optional: depends on how you run the script)
    
    class SAMDataModule(FarmTrackDataModule):
        def train_dataloader(self):
            return DataLoader(
                AgVisionSAMDataset(self.ds, self.train_idx, self.mask_type), 
                batch_size=8, 
                shuffle=True, 
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )
        def val_dataloader(self):
            return DataLoader(
                AgVisionSAMDataset(self.ds, self.val_idx, self.mask_type), 
                batch_size=8, 
                shuffle=False, 
                num_workers=4,
                pin_memory=True,
                persistent_workers=True
            )

    os.makedirs("models/weights", exist_ok=True)
    module = SAMFarmTrackModule()
    datamodule = SAMDataModule(batch_size=8)
    
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/weights/",
        filename="sam-farmtrack-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback],
        accelerator="cuda" if torch.cuda.is_available() else "auto",
        devices=1,
        precision="16-mixed",
        accumulate_grad_batches=2, # Effective batch size 8 * 2 = 16
        log_every_n_steps=5,
    )
    
    import glob
    checkpoint_dir = "models/weights/"
    checkpoints = glob.glob(os.path.join(checkpoint_dir, "*.ckpt"))
    latest_checkpoint = max(checkpoints, key=os.path.getmtime) if checkpoints else None

    print("🚀 Starting OPTIMIZED CUDA SAM Training...")
    if latest_checkpoint:
        print(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
        trainer.fit(module, datamodule=datamodule, ckpt_path=latest_checkpoint)
    else:
        trainer.fit(module, datamodule=datamodule)
    
    final_path = "models/weights/sam_farmtrack_final.pth"
    torch.save(module.model.state_dict(), final_path)
    print(f"✅ Training Complete! Saved to {final_path}")
