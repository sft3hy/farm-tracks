import torch

def calculate_iou(preds, targets, threshold=0.5):
    """
    Calculates Intersection over Union.
    Args:
        preds: logits or probabilities (B, C, H, W)
        targets: ground truth masks (B, C, H, W)
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum((2, 3))
    union = preds.sum((2, 3)) + targets.sum((2, 3)) - intersection
    
    # Avoid division by zero
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean().item()

def calculate_f1(preds, targets, threshold=0.5):
    """
    Calculates F1 Score / Dice Coefficient.
    """
    preds = (torch.sigmoid(preds) > threshold).float()
    intersection = (preds * targets).sum((2, 3))
    
    f1 = (2 * intersection + 1e-6) / (preds.sum((2, 3)) + targets.sum((2, 3)) + 1e-6)
    return f1.mean().item()

class Evaluator:
    def __init__(self):
        self.ious = []
        self.f1s = []

    def update(self, preds, targets):
        self.ious.append(calculate_iou(preds, targets))
        self.f1s.append(calculate_f1(preds, targets))
        
    def compute(self):
        return {
            "mIoU": sum(self.ious) / len(self.ious),
            "mF1": sum(self.f1s) / len(self.f1s)
        }
