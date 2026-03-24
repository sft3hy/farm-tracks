import torch
import torch.nn as nn
import segmentation_models_pytorch as smp

class UNetFarmTrack(nn.Module):
    """
    U-Net baseline with ResNet34 encoder.
    """
    def __init__(self, encoder_name="resnet34", in_channels=3, classes=1):
        super(UNetFarmTrack, self).__init__()
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights="imagenet",
            in_channels=in_channels,
            classes=classes
        )

    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = UNetFarmTrack()
    x = torch.randn(1, 3, 512, 512)
    print("U-Net Output Shape:", model(x).shape)
