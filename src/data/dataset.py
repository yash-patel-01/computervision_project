"""
Dataset class for football tracking with bounding boxes.
Expects annotations in COCO format or custom format with bounding box coordinates.
"""

import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class FootballTrackingDataset(Dataset):
    """
    Dataset for football player and ball tracking.
    
    Expected annotation format (JSON):
    {
        "images": [
            {
                "id": 1,
                "file_name": "frame_001.jpg",
                "width": 1920,
                "height": 1080
            }
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,  # 1: player, 2: ball
                "bbox": [x, y, width, height],  # COCO format
                "area": width * height,
                "iscrowd": 0
            }
        ],
        "categories": [
            {"id": 1, "name": "player"},
            {"id": 2, "name": "ball"}
        ]
    }
    """
    
    def __init__(self, root_dir, annotation_file, transforms=None):
        """
        Args:
            root_dir (str): Directory containing the images
            annotation_file (str): Path to annotation file (JSON)
            transforms (albumentations.Compose): Optional transforms
        """
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Load annotations
        with open(annotation_file, 'r') as f:
            self.coco_data = json.load(f)
        
        self.images = {img['id']: img for img in self.coco_data['images']}
        self.categories = {cat['id']: cat['name'] for cat in self.coco_data['categories']}
        
        # Group annotations by image
        self.image_annotations = {}
        for ann in self.coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_annotations:
                self.image_annotations[img_id] = []
            self.image_annotations[img_id].append(ann)
        
        self.image_ids = list(self.images.keys())
    
    def __len__(self):
        return len(self.image_ids)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (tensor): Image tensor
            target (dict): Dictionary containing:
                - boxes (tensor): Bounding boxes [N, 4] in [x_min, y_min, x_max, y_max] format
                - labels (tensor): Class labels [N]
                - image_id (tensor): Image ID
                - area (tensor): Area of bounding boxes [N]
                - iscrowd (tensor): Is crowd flag [N]
        """
        image_id = self.image_ids[idx]
        image_info = self.images[image_id]
        
        # Load image
        img_path = os.path.join(self.root_dir, image_info['file_name'])
        image = Image.open(img_path).convert('RGB')
        image = np.array(image)
        
        # Get annotations for this image
        annotations = self.image_annotations.get(image_id, [])
        
        boxes = []
        labels = []
        areas = []
        iscrowd = []
        
        for ann in annotations:
            # COCO format: [x, y, width, height]
            x, y, w, h = ann['bbox']
            # Convert to [x_min, y_min, x_max, y_max]
            boxes.append([x, y, x + w, y + h])
            labels.append(ann['category_id'])
            areas.append(ann.get('area', w * h))
            iscrowd.append(ann.get('iscrowd', 0))
        
        # Handle empty annotations
        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            areas = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)
            areas = torch.as_tensor(areas, dtype=torch.float32)
            iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)
        
        target = {
            'boxes': boxes,
            'labels': labels,
            'image_id': torch.tensor([image_id]),
            'area': areas,
            'iscrowd': iscrowd
        }
        
        # Apply transforms
        if self.transforms:
            # Albumentations format
            transformed = self.transforms(
                image=image,
                bboxes=boxes.numpy() if len(boxes) > 0 else [],
                labels=labels.numpy() if len(labels) > 0 else []
            )
            image = transformed['image']
            if len(transformed['bboxes']) > 0:
                target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
        else:
            # Convert to tensor
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        
        return image, target


def get_train_transforms():
    """Get training data augmentation transforms."""
    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Blur(blur_limit=3, p=0.1),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_val_transforms():
    """Get validation transforms (no augmentation)."""
    return A.Compose([
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    """
    Custom collate function for DataLoader.
    Handles variable number of objects per image.
    """
    return tuple(zip(*batch))
