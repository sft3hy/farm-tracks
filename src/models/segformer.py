import torch
import torch.nn as nn
from transformers import SegformerForSemanticSegmentation

class SegformerFarmTrack(nn.Module):
    """
    SegFormer model wrapped for farm track segmentation.
    """
    def __init__(self, pretrained_model_name="nvidia/mit-b0", num_classes=1):
        super(SegformerFarmTrack, self).__init__()
        # We set num_labels based on binary classification (background vs track)
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            pretrained_model_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        outputs = self.model(pixel_values=x)
        logits = outputs.logits
        # Upsample logits to original image size
        upsampled_logits = nn.functional.interpolate(
            logits, size=x.shape[-2:], mode="bilinear", align_corners=False
        )
        return upsampled_logits

if __name__ == "__main__":
    model = SegformerFarmTrack()
    x = torch.randn(1, 3, 512, 512)
    print("SegFormer Output Shape:", model(x).shape)
