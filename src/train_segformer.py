import pytorch_lightning as pl
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Image as HFImage, Value
from models.segformer import SegformerFarmTrack
import os

from data.ag_vision import FarmTrackDataModule, AgVisionDictDataset


class SegformerModule(pl.LightningModule):
    def __init__(self, learning_rate=6e-5):
        super().__init__()
        self.model = SegformerFarmTrack()
        self.learning_rate = learning_rate
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("val_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)


if __name__ == "__main__":
    os.makedirs("models/weights", exist_ok=True)
    module = SegformerModule()
    datamodule = FarmTrackDataModule(batch_size=4)  # Smaller batch for T4 optimization

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/weights/",
        filename="segformer-farmtrack-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )

    trainer = pl.Trainer(
        max_epochs=15,
        callbacks=[checkpoint_callback],
        accelerator="cuda" if torch.cuda.is_available() else "auto",
        devices=1,
        precision="16-mixed",  # T4 optimization
    )

    print("🚀 Starting CUDA SegFormer Training...")
    trainer.fit(module, datamodule=datamodule)

    final_path = "models/weights/segformer_farmtrack_final.pth"
    torch.save(module.model.state_dict(), final_path)
    print(f"✅ Training Complete! Saved to {final_path}")
