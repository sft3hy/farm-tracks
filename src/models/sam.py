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

    def forward(self, input_images, input_points=None, input_labels=None):
        # SAM processor requires inputs on CPU for preprocessing
        input_images = to_cpu(input_images)
        input_points = to_cpu(input_points) if input_points is not None else None
        input_labels = to_cpu(input_labels) if input_labels is not None else None

        # Handle batching for points and labels: SamProcessor expects
        # input_points as list[list[list[float]]] (batch x num_points x 2)
        # input_labels as list[list[int]] (batch x num_points)
        # We must convert numpy arrays to pure Python nested lists.
        if isinstance(input_points, np.ndarray):
            input_points = input_points.tolist()
        elif isinstance(input_points, list):
            input_points = [p.tolist() if isinstance(p, np.ndarray) else p for p in input_points]

        if isinstance(input_labels, np.ndarray):
            input_labels = input_labels.tolist()
        elif isinstance(input_labels, list):
            input_labels = [l.tolist() if isinstance(l, np.ndarray) else l for l in input_labels]

        # SamProcessor expects points as [batch[points_per_image[point]]]
        # i.e. each element should be a list of [x, y] pairs wrapped in another list
        # Shape check: if points are (B, N, 2) -> already correct nesting
        # If points are flat list of [x, y] pairs (no batch dim), wrap in list
        if input_points is not None and len(input_points) > 0:
            # Ensure 3-level nesting: batch -> points_set -> point_coords
            if isinstance(input_points[0], (int, float)):
                # Single point [x, y] -> wrap twice
                input_points = [[input_points]]
            elif isinstance(input_points[0], list) and len(input_points[0]) > 0 and isinstance(input_points[0][0], (int, float)):
                # List of points [[x,y], ...] -> could be single-image multi-point or batch of single points
                # Check if inner elements are 2-element coords
                if len(input_points[0]) == 2 and not isinstance(input_points[0][0], list):
                    # Ambiguous, but default: treat as batch of single points
                    input_points = [[p] for p in input_points]

        # inputs can be pre-processed by SamProcessor
        inputs = self.processor(
            input_images, 
            input_points=input_points, 
            input_labels=input_labels,
            return_tensors="pt"
        )
        
        # Move to model device
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        outputs = self.model(**inputs)
        
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
    # Note: SAM requires proper processor usage; this is a skeleton structure.
    pass
