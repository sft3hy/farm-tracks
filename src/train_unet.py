import pytorch_lightning as pl
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Image as HFImage, Value


from data.ag_vision import FarmTrackDataModule, AgVisionDictDataset


class FarmTrackModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = DiceLoss(mode="binary", from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch["image"], batch["mask"]
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        return [optimizer], [scheduler]


if __name__ == "__main__":
    from models.unet import UNetFarmTrack
    import os

    # Instantiate the model
    model = UNetFarmTrack()
    module = FarmTrackModule(model)
    datamodule = FarmTrackDataModule(
        batch_size=8, mask_type="planter_skip"
    )  # using planter skip as the "track" metric

    # Save checkpoints to the models/weights directory
    os.makedirs("models/weights", exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="models/weights/",
        filename="unet-farmtrack-{epoch:02d}-{val_loss:.2f}",
        save_top_k=1,
        monitor="val_loss",
    )

    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs=10,
        callbacks=[checkpoint_callback],
        accelerator="auto",  # automatically picks GPU/MPS if available
        devices=1,
    )

    print("Initiating PyTorch Lightning Training Protocol...")
    trainer.fit(module, datamodule=datamodule)

    # Save the final compiled model weights for Gradio!
    final_path = "models/weights/unet_farmtrack_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete! Compiled weights successfully saved to {final_path}")
