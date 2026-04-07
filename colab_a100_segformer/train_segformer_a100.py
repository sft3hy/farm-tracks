import os
import shutil
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from datasets import load_dataset, Features, Image as HFImage, Value
from transformers import SegformerForSemanticSegmentation
from segmentation_models_pytorch.losses import DiceLoss

# ==========================================
# Google Drive Persistent Storage
# ==========================================
# Auto-detect mounted Google Drive for checkpoint persistence.
# If Drive is mounted, all weights are saved there so they survive runtime disconnects.
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/farm-tracks/models/segformer"
if os.path.isdir("/content/drive/MyDrive"):
    OUTPUT_DIR = DRIVE_OUTPUT_DIR
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"💾 Google Drive detected — saving weights to: {OUTPUT_DIR}")
else:
    OUTPUT_DIR = "models/weights"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"⚠️  Google Drive not mounted — saving weights locally to: {OUTPUT_DIR}")
    print(f"   To persist weights, mount Drive before training.")

# ==========================================
# A100 Full-Utilization Optimizations
# ==========================================
# Enable TF32 for matrix multiplications (A100 native, ~3x faster than FP32)
torch.set_float32_matmul_precision("high")
# Enable cuDNN benchmark mode - auto-tunes convolution algorithms for fixed input sizes
torch.backends.cudnn.benchmark = True


# ==========================================
# 1. Dataset & DataModule
# ==========================================
class AgVisionDictDataset(Dataset):
    def __init__(self, ds, index_list, mask_type="planter_skip"):
        self.ds = ds
        self.index_list = index_list
        self.mask_type = mask_type
        self.transform_rgb = T.Compose(
            [
                T.ToTensor(),
                T.Resize((512, 512), antialias=True),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        self.transform_mask = T.Compose(
            [
                T.ToTensor(),
                T.Resize(
                    (512, 512),
                    antialias=True,
                    interpolation=T.InterpolationMode.NEAREST,
                ),
            ]
        )

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        file_id, mapping = self.index_list[idx]
        rgb_idx = mapping.get("rgb")
        if rgb_idx is None:
            img_tensor = torch.zeros(3, 512, 512)
        else:
            sample = self.ds[rgb_idx]
            img = sample.get("jpg") or sample.get("png")
            img_tensor = self.transform_rgb(img.convert("RGB"))

        mask_idx = mapping.get(self.mask_type)
        if mask_idx is None:
            mask_tensor = torch.zeros(1, 512, 512)
        else:
            mask_sample = self.ds[mask_idx]
            mask_img = mask_sample.get("png")
            mask_tensor = self.transform_mask(mask_img.convert("L"))
            mask_tensor = (mask_tensor > 0.5).float()

        return {"image": img_tensor, "mask": mask_tensor}


class FarmTrackDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=32, mask_type="planter_skip"):
        super().__init__()
        self.batch_size = batch_size
        self.mask_type = mask_type

    def setup(self, stage=None):
        features = Features(
            {
                "png": HFImage(),
                "jpg": HFImage(),
                "__key__": Value("string"),
                "__url__": Value("string"),
            }
        )
        self.ds = load_dataset(
            "shi-labs/Agriculture-Vision", split="train", features=features
        )

        print(f"🔍 Indexing {len(self.ds):,} samples (Speed-Optimized)...")
        all_keys = self.ds["__key__"]

        mapping = {}
        from tqdm import tqdm

        for i, key in enumerate(tqdm(all_keys, desc="  Mapping fleet")):
            if not key:
                continue
            parts = key.split("/")
            file_id, folder = parts[-1], parts[-2] if len(parts) >= 2 else None
            if file_id not in mapping:
                mapping[file_id] = {}
            if folder == "rgb":
                mapping[file_id]["rgb"] = i
            else:
                mapping[file_id][folder] = i

        # Only keep file_ids that have BOTH rgb AND the target mask type
        self.index_list = [
            (k, v) for k, v in mapping.items() if "rgb" in v and self.mask_type in v
        ]
        print(
            f"✅ Found {len(self.index_list):,} paired samples (RGB + {self.mask_type})"
        )

        split_point = int(0.8 * len(self.index_list))
        self.train_idx = self.index_list[:split_point]
        self.val_idx = self.index_list[split_point:]
        print(f"   Train: {len(self.train_idx):,}  |  Val: {len(self.val_idx):,}")

    def train_dataloader(self):
        # Use all CPU cores. prefetch_factor=4 keeps GPU fed while CPU decodes next batches.
        num_workers = os.cpu_count() or 4
        return DataLoader(
            AgVisionDictDataset(self.ds, self.train_idx, self.mask_type),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
            drop_last=True,  # Avoids irregular last-batch slowdowns
        )

    def val_dataloader(self):
        num_workers = os.cpu_count() or 4
        return DataLoader(
            AgVisionDictDataset(self.ds, self.val_idx, self.mask_type),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=4,
        )


# ==========================================
# 2. SegFormer Model
# ==========================================
class SegformerFarmTrack(nn.Module):
    def __init__(self, pretrained_model_name="nvidia/mit-b4", num_classes=1):
        super(SegformerFarmTrack, self).__init__()
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        return nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )


# ==========================================
# 3. Lightning Module — Full A100 Utilization
# ==========================================
class SegformerModule(pl.LightningModule):
    def __init__(self, learning_rate=6e-5, max_epochs=8):
        super().__init__()
        self.model = SegformerFarmTrack()
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("train_loss", loss, prog_bar=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)

        # Compute IoU metric for monitoring convergence
        preds = (logits.sigmoid() > 0.5).float()
        intersection = (preds * y).sum()
        union = preds.sum() + y.sum() - intersection
        iou = (intersection + 1e-6) / (union + 1e-6)

        self.log("val_loss", loss, prog_bar=True, sync_dist=True)
        self.log("val_iou", iou, prog_bar=True, sync_dist=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=0.01
        )
        # OneCycleLR: aggressive warmup → peak → anneal. Converges faster than cosine.
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.learning_rate,
            total_steps=self.trainer.estimated_stepping_batches,
            pct_start=0.1,   # 10% warmup
            anneal_strategy="cos",
            div_factor=10,   # start_lr = max_lr / 10
            final_div_factor=100,  # end_lr = start_lr / 100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }


# ==========================================
# 4. Training Execution — A100 Full Send
# ==========================================
if __name__ == "__main__":
    # ---- Tunable Knobs ----
    BATCH_SIZE = 16       # 16 × 512² bf16 ≈ 28-30 GB VRAM (SegFormer lacks grad checkpoint)
    MAX_EPOCHS = 8        # OneCycleLR + early stopping; 8 is plenty for convergence
    GRAD_ACCUM = 4        # Effective batch = 64. Fills GPU while staying under 40GB
    LEARNING_RATE = 6e-5  # Slightly higher LR matched to larger effective batch

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Log the plan
    print("=" * 60)
    print("⚡ A100 FULL-UTILIZATION TRAINING CONFIG")
    print("=" * 60)
    print(f"  Batch size (per step):    {BATCH_SIZE}")
    print(f"  Gradient accumulation:    {GRAD_ACCUM}")
    print(f"  Effective batch size:     {BATCH_SIZE * GRAD_ACCUM}")
    print(f"  Steps/epoch (approx):     ~{60000 // BATCH_SIZE // GRAD_ACCUM}")
    print(f"  Max epochs:               {MAX_EPOCHS}")
    print(f"  Learning rate:            {LEARNING_RATE}")
    print(f"  Precision:                bf16-mixed")
    print(f"  torch.compile:            default (no CUDA graphs)")
    print("=" * 60)

    # 1. Initialize Module
    module = SegformerModule(learning_rate=LEARNING_RATE, max_epochs=MAX_EPOCHS)

    # 2. torch.compile — fuses ops, removes Python overhead, generates optimized Triton kernels.
    #    "default" mode: full op fusion + Triton codegen WITHOUT CUDA graph capture.
    #    (max-autotune adds CUDA graphs which duplicate the entire model in VRAM — causes OOM)
    module.model = torch.compile(module.model, mode="default")

    # 3. DataModule
    datamodule = FarmTrackDataModule(batch_size=BATCH_SIZE)

    # 4. Callbacks
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=OUTPUT_DIR,
        filename="segformer-b4-farmtrack-{epoch:02d}-{val_loss:.2f}",
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        every_n_epochs=1,  # Ensure at least one checkpoint per epoch on Drive
    )
    early_stop_callback = pl.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, mode="min"  # Tighter patience with fewer epochs
    )
    lr_monitor = pl.callbacks.LearningRateMonitor(logging_interval="step")

    # 5. Trainer
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        callbacks=[checkpoint_callback, early_stop_callback, lr_monitor],
        accelerator="cuda",
        devices=1,
        precision="bf16-mixed",             # BF16: native A100, no grad scaling needed
        accumulate_grad_batches=GRAD_ACCUM,  # Effective batch = BATCH_SIZE × GRAD_ACCUM
        gradient_clip_val=1.0,               # Prevents gradient explosion with high LR
        log_every_n_steps=10,
        val_check_interval=0.5,              # Validate twice per epoch (catch improvements early)
        deterministic=False,                 # Allow non-deterministic for speed
    )

    print("🚀 Starting SegFormer (mit-b4) A100 Full-Utilization Training...")
    trainer.fit(module, datamodule=datamodule)

    # Save final model to persistent storage
    final_path = os.path.join(OUTPUT_DIR, "segformer_b4_farmtrack_final.pth")
    # Unwrap compiled model for clean state_dict save
    raw_model = module.model._orig_mod if hasattr(module.model, "_orig_mod") else module.model
    torch.save(raw_model.state_dict(), final_path)
    print(f"✅ Training Complete! Saved to {final_path}")

    # Also save the best checkpoint path for reference
    best_ckpt = checkpoint_callback.best_model_path
    if best_ckpt:
        print(f"🏆 Best checkpoint (val_loss={checkpoint_callback.best_model_score:.4f}): {best_ckpt}")

    # List all saved files
    print(f"\n📁 Files in {OUTPUT_DIR}:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1e6
        print(f"   {f}  ({size_mb:.1f} MB)")
