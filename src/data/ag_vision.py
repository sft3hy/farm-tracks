import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset, Features, Image as HFImage, Value
from transformers import SamProcessor
import numpy as np

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

class AgVisionSAMDataset(Dataset):
    def __init__(self, ds, index_list, mask_type="planter_skip"):
        self.ds = ds
        self.index_list = index_list
        self.mask_type = mask_type
        # Efficient preprocessing on CPU
        self.processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

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

        # SAM Preprocessing
        input_point = [[[256, 256]]] 
        input_label = [[1]]
        
        inputs = self.processor(
            image, 
            input_points=input_point, 
            input_labels=input_label, 
            return_tensors="pt"
        )
        
        return {
            "pixel_values": inputs["pixel_values"].squeeze(0),
            "input_points": inputs["input_points"].squeeze(0),
            "input_labels": inputs["input_labels"].squeeze(0),
            "mask": torch.from_numpy(mask).unsqueeze(0) # (1, H, W)
        }

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
            pin_memory=True,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            AgVisionDictDataset(self.ds, self.val_idx, self.mask_type),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            persistent_workers=True
        )
