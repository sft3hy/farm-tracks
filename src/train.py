import pytorch_lightning as pl
import torch
import torch.nn as nn
from segmentation_models_pytorch.losses import DiceLoss
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Image as HFImage, Value

class AgVisionDictDataset(Dataset):
    def __init__(self, ds, index_list, mask_type='planter_skip'):
        self.ds = ds
        self.index_list = index_list
        self.mask_type = mask_type
        self.transform_rgb = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.transform_mask = T.Compose([
            T.ToTensor(),
            T.Resize((512, 512), antialias=True, interpolation=T.InterpolationMode.NEAREST)
        ])
        
    def __len__(self):
        return len(self.index_list)
        
    def __getitem__(self, idx):
        file_id, mapping = self.index_list[idx]
        
        # Load RGB image
        rgb_idx = mapping.get('rgb')
        if rgb_idx is None:
            img_tensor = torch.zeros(3, 512, 512)
        else:
            sample = self.ds[rgb_idx]
            img = sample.get('jpg') or sample.get('png')
            img_tensor = self.transform_rgb(img.convert('RGB'))
            
        # Load Mask (using planter_skip as it best approximates track anomalies)
        mask_idx = mapping.get(self.mask_type)
        if mask_idx is None:
            mask_tensor = torch.zeros(1, 512, 512)
        else:
            mask_sample = self.ds[mask_idx]
            mask_img = mask_sample.get('png')
            mask_tensor = self.transform_mask(mask_img.convert('L'))
            mask_tensor = (mask_tensor > 0.5).float()
            
        return {'image': img_tensor, 'mask': mask_tensor}

class FarmTrackDataModule(pl.LightningDataModule):
    def __init__(self, batch_size=8, mask_type='planter_skip'):
        super().__init__()
        self.batch_size = batch_size
        self.mask_type = mask_type
        
    def setup(self, stage=None):
        features = Features({
            'png': HFImage(),
            'jpg': HFImage(),
            '__key__': Value('string'),
            '__url__': Value('string')
        })
        print("Loading HF dataset...")
        # Accessing the exact same arrow cache the Gradio app built
        self.ds = load_dataset(
            "shi-labs/Agriculture-Vision", 
            split="train", 
            features=features
        )
        
        print("Building index mapping for dataset pairs...")
        mapping = {}
        for i, sample in enumerate(self.ds):
            key = sample.get('__key__', '')
            if not key: continue
            
            parts = key.split('/')
            file_id = parts[-1]
            folder = parts[-2] if len(parts) >= 2 else None
            
            if file_id not in mapping:
                mapping[file_id] = {}
                
            if folder == 'rgb':
                mapping[file_id]['rgb'] = i
            else:
                mapping[file_id][folder] = i
                
        # Filter out anything that doesn't have an RGB image
        self.index_list = [(k, v) for k, v in mapping.items() if 'rgb' in v]
        print(f"Found {len(self.index_list)} usable agricultural fields.")
        
        # 80/20 train/val split
        split_point = int(0.8 * len(self.index_list))
        self.train_idx = self.index_list[:split_point]
        self.val_idx = self.index_list[split_point:]
        
    def train_dataloader(self):
        ds = AgVisionDictDataset(self.ds, self.train_idx, self.mask_type)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self):
        ds = AgVisionDictDataset(self.ds, self.val_idx, self.mask_type)
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False, num_workers=4)

class FarmTrackModule(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.loss_fn = DiceLoss(mode='binary', from_logits=True)
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['mask']
        logits = self(x)
        loss = self.loss_fn(logits, y) + self.bce_loss(logits, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
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
    datamodule = FarmTrackDataModule(batch_size=8, mask_type='planter_skip') # using planter skip as the "track" metric
    
    # Save checkpoints to the models/weights directory
    os.makedirs('models/weights', exist_ok=True)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath='models/weights/',
        filename='unet-farmtrack-{epoch:02d}-{val_loss:.2f}',
        save_top_k=1,
        monitor='val_loss'
    )
    
    # Setup Trainer
    trainer = pl.Trainer(
        max_epochs=10, 
        callbacks=[checkpoint_callback],
        accelerator='auto', # automatically picks GPU/MPS if available
        devices=1
    )
    
    print("Initiating PyTorch Lightning Training Protocol...")
    trainer.fit(module, datamodule=datamodule)
    
    # Save the final compiled model weights for Gradio!
    final_path = "models/weights/unet_farmtrack_final.pth"
    torch.save(model.state_dict(), final_path)
    print(f"Training Complete! Compiled weights successfully saved to {final_path}")
