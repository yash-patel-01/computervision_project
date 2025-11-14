"""
Object detection model for football tracking.
Uses Faster R-CNN with ResNet50 backbone from torchvision.
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def create_model(num_classes, pretrained=True, trainable_backbone_layers=3):
    """
    Create a Faster R-CNN model for object detection.
    
    Args:
        num_classes (int): Number of classes (including background)
        pretrained (bool): Use pretrained weights on COCO
        trainable_backbone_layers (int): Number of trainable backbone layers (0-5)
    
    Returns:
        model: Faster R-CNN model
    """
    # Load pre-trained model
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers
    )
    
    # Replace the classifier head
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


class FootballDetector:
    """
    Wrapper class for the football detection model.
    Handles training and inference.
    """
    
    def __init__(self, num_classes=3, device='cuda'):
        """
        Initialize the detector.
        
        Args:
            num_classes (int): Number of classes including background
                              (background=0, player=1, ball=2)
            device (str): Device to run the model on ('cuda' or 'cpu')
        """
        self.num_classes = num_classes
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = create_model(num_classes).to(self.device)
        self.model.train()
    
    def train_mode(self):
        """Set model to training mode."""
        self.model.train()
    
    def eval_mode(self):
        """Set model to evaluation mode."""
        self.model.eval()
    
    def forward(self, images, targets=None):
        """
        Forward pass through the model.
        
        Args:
            images (list of tensors): List of images
            targets (list of dicts): List of targets (for training)
        
        Returns:
            During training: dict with losses
            During inference: list of dicts with predictions
        """
        return self.model(images, targets)
    
    def predict(self, images, confidence_threshold=0.5):
        """
        Make predictions on images.
        
        Args:
            images (list of tensors): List of images
            confidence_threshold (float): Minimum confidence score
        
        Returns:
            predictions (list of dicts): Filtered predictions
        """
        self.eval_mode()
        with torch.no_grad():
            predictions = self.model(images)
        
        # Filter by confidence threshold
        filtered_predictions = []
        for pred in predictions:
            keep = pred['scores'] > confidence_threshold
            filtered_pred = {
                'boxes': pred['boxes'][keep],
                'labels': pred['labels'][keep],
                'scores': pred['scores'][keep]
            }
            filtered_predictions.append(filtered_pred)
        
        return filtered_predictions
    
    def save(self, path):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
    
    def load(self, path):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.num_classes = checkpoint['num_classes']
