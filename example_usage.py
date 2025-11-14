"""
Example script demonstrating how to use the football tracking model.
This script shows:
1. How to train the model
2. How to run inference on images and videos
3. How to visualize results
"""

import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

import torch
from models.detector import FootballDetector
from data.dataset import FootballTrackingDataset, get_train_transforms, collate_fn
from torch.utils.data import DataLoader


def example_training():
    """
    Example of how to train the model.
    
    Run this from command line:
    python src/train.py \
        --train-data-dir data/train/images \
        --train-annotations data/train/annotations.json \
        --val-data-dir data/val/images \
        --val-annotations data/val/annotations.json \
        --batch-size 4 \
        --epochs 50 \
        --checkpoint-dir checkpoints \
        --log-dir runs
    """
    print("=" * 60)
    print("TRAINING EXAMPLE")
    print("=" * 60)
    print("\nTo train the model, run:")
    print("\npython src/train.py \\")
    print("    --train-data-dir data/train/images \\")
    print("    --train-annotations data/train/annotations.json \\")
    print("    --val-data-dir data/val/images \\")
    print("    --val-annotations data/val/annotations.json \\")
    print("    --batch-size 4 \\")
    print("    --epochs 50 \\")
    print("    --checkpoint-dir checkpoints \\")
    print("    --log-dir runs")
    print("\nOptional: Use a config file:")
    print("python src/train.py --config configs/default_config.yaml")
    print()


def example_inference_image():
    """
    Example of how to run inference on a single image.
    
    Run this from command line:
    python src/inference.py \
        --checkpoint checkpoints/best_model.pth \
        --image path/to/image.jpg \
        --output output_image.jpg \
        --confidence-threshold 0.5
    """
    print("=" * 60)
    print("IMAGE INFERENCE EXAMPLE")
    print("=" * 60)
    print("\nTo run inference on a single image:")
    print("\npython src/inference.py \\")
    print("    --checkpoint checkpoints/best_model.pth \\")
    print("    --image path/to/image.jpg \\")
    print("    --output output_image.jpg \\")
    print("    --confidence-threshold 0.5")
    print()


def example_inference_video():
    """
    Example of how to run inference on a video.
    
    Run this from command line:
    python src/inference.py \
        --checkpoint checkpoints/best_model.pth \
        --video path/to/video.mp4 \
        --output output_video.mp4 \
        --confidence-threshold 0.5 \
        --skip-frames 1
    """
    print("=" * 60)
    print("VIDEO INFERENCE EXAMPLE")
    print("=" * 60)
    print("\nTo run inference on a video:")
    print("\npython src/inference.py \\")
    print("    --checkpoint checkpoints/best_model.pth \\")
    print("    --video path/to/football_clip.mp4 \\")
    print("    --output output_video.mp4 \\")
    print("    --confidence-threshold 0.5 \\")
    print("    --skip-frames 1")
    print("\nNote: Set --skip-frames to a higher value (e.g., 5) for faster processing")
    print()


def example_inference_frames():
    """
    Example of how to run inference on a directory of frames.
    
    Run this from command line:
    python src/inference.py \
        --checkpoint checkpoints/best_model.pth \
        --frames-dir path/to/frames \
        --output path/to/output_frames \
        --confidence-threshold 0.5
    """
    print("=" * 60)
    print("BATCH FRAMES INFERENCE EXAMPLE")
    print("=" * 60)
    print("\nTo run inference on a directory of frames:")
    print("\npython src/inference.py \\")
    print("    --checkpoint checkpoints/best_model.pth \\")
    print("    --frames-dir path/to/frames \\")
    print("    --output path/to/output_frames \\")
    print("    --confidence-threshold 0.5")
    print()


def example_programmatic_usage():
    """
    Example of how to use the model programmatically.
    """
    print("=" * 60)
    print("PROGRAMMATIC USAGE EXAMPLE")
    print("=" * 60)
    print("\nExample Python code for using the model:")
    print()
    
    code = """
import torch
from models.detector import FootballDetector
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Initialize model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = FootballDetector(num_classes=3, device=device)

# Load checkpoint
model.load('checkpoints/best_model.pth')
model.eval_mode()

# Prepare image
image = Image.open('football_frame.jpg').convert('RGB')
image_np = np.array(image)

# Apply transforms
transforms = A.Compose([
    A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ToTensorV2()
])
transformed = transforms(image=image_np)
image_tensor = transformed['image'].to(device)

# Run inference
predictions = model.predict([image_tensor], confidence_threshold=0.5)[0]

# Print results
print(f"Detected {len(predictions['boxes'])} objects")
for i, (box, label, score) in enumerate(zip(
    predictions['boxes'], 
    predictions['labels'], 
    predictions['scores']
)):
    class_name = 'player' if label == 1 else 'ball'
    print(f"Object {i}: {class_name} with confidence {score:.2f}")
    print(f"  Bounding box: {box.tolist()}")
"""
    
    print(code)


def show_annotation_format():
    """
    Show the expected annotation format.
    """
    print("=" * 60)
    print("ANNOTATION FORMAT")
    print("=" * 60)
    print("\nThe model expects annotations in COCO format (JSON):")
    print()
    
    format_example = """{
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
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": width * height,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "player"},
    {"id": 2, "name": "ball"}
  ]
}

Notes:
- bbox format: [x, y, width, height] (COCO format)
- category_id: 1 for player, 2 for ball
- Images should be placed in a directory, with paths relative to that directory
- See data/train/annotations.json for a complete example
"""
    
    print(format_example)


def main():
    """Main function showing all examples."""
    print("\n" + "=" * 60)
    print("FOOTBALL TRACKING MODEL - USAGE EXAMPLES")
    print("=" * 60)
    print("\nThis script demonstrates how to use the football tracking model")
    print("for detecting and tracking players and the ball in football clips.")
    print()
    
    # Show all examples
    example_training()
    example_inference_image()
    example_inference_video()
    example_inference_frames()
    example_programmatic_usage()
    show_annotation_format()
    
    print("=" * 60)
    print("For more information, see README.md")
    print("=" * 60)
    print()


if __name__ == '__main__':
    main()
