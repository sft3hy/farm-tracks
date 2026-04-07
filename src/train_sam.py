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

# Optimize CUDA memory allocation to prevent fragmentation if available
if torch.cuda.is_available():
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# NOTE: PYTORCH_MPS_HIGH_WATERMARK_RATIO is intentionally NOT set here.
# Any fractional value (e.g. 0.75) causes a crash because MPS internally
# validates that the low watermark ratio (hardcoded default: 1.4) does not
# exceed the high watermark — which it always will for values < 1.0.
# Memory is controlled instead via smaller batch size and fewer DataLoader workers.

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
            logits = nn.functional.interpolate(
                logits, size=y.shape[-2:], mode="bilinear"
            )

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
            logits = nn.functional.interpolate(
                logits, size=y.shape[-2:], mode="bilinear"
            )

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
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
                pin_memory=False,
                # Keep workers alive between epochs — PL warned about this overhead
                persistent_workers=True,
                prefetch_factor=4,      # Feed GPU ahead of time
            )

        def val_dataloader(self):
            return DataLoader(
                AgVisionSAMDataset(self.ds, self.val_idx, self.mask_type),
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=False,
                persistent_workers=True,
                prefetch_factor=4,
            )

    os.makedirs("models/weights", exist_ok=True)
    module = SAMFarmTrackModule()
    # batch_size=4: same effective batch as before (4 * 3 = 12), but half the
    # forward passes per epoch vs batch=2 * accumulate=6. RAM is fine at this size.
    datamodule = SAMDataModule(batch_size=4)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/weights/",
        filename="sam-farmtrack-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )

    # Determine best device
    if torch.cuda.is_available():
        accelerator = "cuda"
        precision = "16-mixed"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        # bf16 is more numerically stable than fp16 on Apple Silicon and avoids
        # the NaN spikes that can occur with 16-mixed on MPS.
        precision = "bf16-mixed"
    else:
        accelerator = "cpu"
        precision = "32"

    trainer = pl.Trainer(
        max_epochs=5,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        precision=precision,
        # Effective batch = 4 * 3 = 12 (unchanged). Halving accumulate steps cuts
        # total forward passes per epoch roughly in half vs batch=2 + accumulate=6.
        accumulate_grad_batches=3,
        # Only validate on 25% of val set per epoch — enough for reliable checkpointing,
        # saves ~3 hours per 2 remaining epochs without touching training params.
        limit_val_batches=0.25,
        # Reduce logging overhead: log every 20 steps instead of every 5
        log_every_n_steps=20,
        gradient_clip_val=1.0,
    )

    import glob

    # Search for checkpoints in both models/weights and lightning_logs
    checkpoint_paths = [
        "models/weights/*.ckpt",
        "lightning_logs/version_*/checkpoints/*.ckpt",
    ]
    checkpoints = []
    for path in checkpoint_paths:
        checkpoints.extend(glob.glob(path))

    # Allow manual override via env var
    env_checkpoint = os.environ.get("SAM_CHECKPOINT")
    if env_checkpoint and os.path.exists(env_checkpoint):
        latest_checkpoint = env_checkpoint
    else:
        latest_checkpoint = (
            max(checkpoints, key=os.path.getmtime) if checkpoints else None
        )

    print(
        f"🚀 Starting SAM Training on {accelerator.upper()} (Precision: {precision})..."
    )
    if latest_checkpoint:
        print(f"🔄 Resuming from checkpoint: {latest_checkpoint}")
        trainer.fit(module, datamodule=datamodule, ckpt_path=latest_checkpoint)
    else:
        trainer.fit(module, datamodule=datamodule)

    final_path = "models/weights/sam_farmtrack_final.pth"
    torch.save(module.model.state_dict(), final_path)
    print(f"✅ Training Complete! Saved to {final_path}")
