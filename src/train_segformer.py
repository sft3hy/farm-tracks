import pytorch_lightning as pl
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Image as HFImage, Value
from models.segformer import SegformerFarmTrack
import os

# Reuse AgVisionDictDataset and FarmTrackDataModule logic from train.py
# (Wait, maybe I should extract them to a common data file?
# The user asked for 3 NEW python scripts, so I will define them here to be self-contained or import if possible.)


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
    def __init__(self, batch_size=4, mask_type="planter_skip"):
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

        mapping = {}
        for i, sample in enumerate(self.ds):
            key = sample.get("__key__", "")
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

        self.index_list = [(k, v) for k, v in mapping.items() if "rgb" in v]
        split_point = int(0.8 * len(self.index_list))
        self.train_idx = self.index_list[:split_point]
        self.val_idx = self.index_list[split_point:]

    def train_dataloader(self):
        return DataLoader(
            AgVisionDictDataset(self.ds, self.train_idx, self.mask_type),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
        )

    def val_dataloader(self):
        return DataLoader(
            AgVisionDictDataset(self.ds, self.val_idx, self.mask_type),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
        )


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
