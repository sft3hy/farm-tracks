import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor
import numpy as np

def to_cpu(data):
    """Recursively moves all tensors to CPU/Numpy for SAM processor compatibility."""
    if torch.is_tensor(data):
        return data.detach().cpu().numpy()
    elif isinstance(data, list):
        return [to_cpu(x) for x in data]
    elif isinstance(data, dict):
        return {k: to_cpu(v) for k, v in data.items()}
    return data

class SAMFarmTrack(nn.Module):
    """
    Adapter for Segment Anything Model (SAM) zero-shot segmentation.
    While SAM is primarily interactive, we can fine-tune its mask decoder,
    or use it with auto-prompting for zero-shot track extraction.
    """
    def __init__(self, model_id="facebook/sam-vit-base"):
        super(SAMFarmTrack, self).__init__()
        self.processor = SamProcessor.from_pretrained(model_id)
        self.model = SamModel.from_pretrained(model_id)
        
        # Freeze vision encoder and prompt encoder for adapter fine-tuning
        for param in self.model.vision_encoder.parameters():
            param.requires_grad = False
        for param in self.model.prompt_encoder.parameters():
            param.requires_grad = False

    def forward(self, input_images, input_points=None):
        # SAM processor requires inputs on CPU for preprocessing
        input_images = to_cpu(input_images)
        input_points = to_cpu(input_points) if input_points is not None else None

        # inputs can be pre-processed by SamProcessor
        inputs = self.processor(input_images, input_points=input_points, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)
        return outputs.pred_masks
        
if __name__ == "__main__":
    # Note: SAM requires proper processor usage; this is a skeleton structure.
    pass
