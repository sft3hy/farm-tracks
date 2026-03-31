import torch
import torch.nn as nn
from transformers import SamModel, SamProcessor
import numpy as np

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

    def forward(self, pixel_values, input_points=None, input_labels=None):
        """
        Expects:
        - pixel_values: (B, C, H, W) tensor OR a PIL Image/list of Images.
        - input_points: (B, P, N, 2) or (B, N, 2) tensor.
        - input_labels: (B, P, N) or (B, N) tensor.
        """
        device = next(self.model.parameters()).device

        # If we got a PIL image (e.g. from server.py), run processor here as a convenience
        if not isinstance(pixel_values, torch.Tensor):
            # SAM Processor expects points as lists when processing raw images
            # But the server passes tensors. We'll convert back to lists for processor if needed,
            # or just rely on the processor's flexibility.
            inputs = self.processor(
                pixel_values, 
                input_points=input_points.cpu().tolist() if torch.is_tensor(input_points) else input_points,
                input_labels=input_labels.cpu().tolist() if torch.is_tensor(input_labels) else input_labels,
                return_tensors="pt"
            )
            pixel_values = inputs["pixel_values"].to(device)
            input_points = inputs["input_points"].to(device)
            input_labels = inputs["input_labels"].to(device)
        
        # Ensure tensors are on correct device
        pixel_values = pixel_values.to(device)
        if input_points is not None:
            input_points = input_points.to(device)
            # Ensure 4D: (batch_size, point_batch_size, nb_points_per_image, 2)
            if input_points.dim() == 3:
                input_points = input_points.unsqueeze(1)
        if input_labels is not None:
            input_labels = input_labels.to(device)
            # Ensure 3D: (batch_size, point_batch_size, nb_points_per_image)
            if input_labels.dim() == 2:
                input_labels = input_labels.unsqueeze(1)

        outputs = self.model(
            pixel_values=pixel_values,
            input_points=input_points,
            input_labels=input_labels,
            return_dict=True
        )
        
        # outputs.pred_masks shape: (batch_size, num_prompts, num_masks, height, width)
        # Squeeze num_prompts dimension if it's 1
        masks = outputs.pred_masks
        if masks.dim() == 5 and masks.shape[1] == 1:
            masks = masks.squeeze(1)
            
        # For simplicity in this baseline, we return the first mask (usually the most confident)
        # Result shape: (batch_size, 1, height, width)
        if masks.dim() == 4 and masks.shape[1] > 1:
            masks = masks[:, 0:1, :, :]
            
        return masks
        
if __name__ == "__main__":
    pass
