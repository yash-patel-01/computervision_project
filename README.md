# Football Tracking Computer Vision Model

A deep learning computer vision model for tracking players and footballs in football clips using bounding box detection. This project uses a Faster R-CNN model with ResNet50 backbone, pre-trained on COCO and fine-tuned on football footage.

## Features

- **Object Detection**: Detects and tracks players and footballs in video frames
- **Bounding Box Tracking**: Provides accurate bounding box coordinates for each detected object
- **Real-time Inference**: Supports both image and video inference
- **Pre-trained Model**: Built on top of Faster R-CNN with transfer learning
- **Easy Training**: Simple training pipeline with data augmentation
- **Visualization**: Built-in tools for visualizing predictions

## Project Structure

```
computervision_project/
├── src/
│   ├── data/
│   │   └── dataset.py          # Dataset class and data loaders
│   ├── models/
│   │   └── detector.py         # Model definition and wrapper
│   ├── utils/
│   │   └── visualization.py    # Visualization utilities
│   ├── train.py                # Training script
│   └── inference.py            # Inference script
├── configs/
│   └── default_config.yaml     # Configuration file
├── data/
│   ├── train/                  # Training data and annotations
│   └── val/                    # Validation data and annotations
├── checkpoints/                # Model checkpoints (created during training)
├── requirements.txt            # Python dependencies
└── example_usage.py           # Example usage scripts
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yash-patel-01/computervision_project.git
cd computervision_project
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python example_usage.py
```

## Data Format

The model expects annotations in COCO format (JSON). Each annotation file should contain:

```json
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
      "category_id": 1,
      "bbox": [x, y, width, height],
      "area": 45000,
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 1, "name": "player"},
    {"id": 2, "name": "ball"}
  ]
}
```

**Categories:**
- `category_id: 1` - Player
- `category_id: 2` - Ball

**Bounding Box Format:**
- `[x, y, width, height]` in COCO format
- `x, y`: Top-left corner coordinates
- `width, height`: Box dimensions

See `data/train/annotations.json` for a complete example.

## Usage

### Training

Train the model on your annotated football clips:

```bash
python src/train.py \
    --train-data-dir data/train/images \
    --train-annotations data/train/annotations.json \
    --val-data-dir data/val/images \
    --val-annotations data/val/annotations.json \
    --batch-size 4 \
    --epochs 50 \
    --checkpoint-dir checkpoints \
    --log-dir runs
```

**Training Parameters:**
- `--batch-size`: Batch size for training (default: 4)
- `--epochs`: Number of training epochs (default: 50)
- `--learning-rate`: Initial learning rate (default: 0.005)
- `--num-workers`: Number of data loading workers (default: 4)

You can also use a configuration file:

```bash
python src/train.py --config configs/default_config.yaml
```

### Inference

#### Single Image Inference

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/image.jpg \
    --output output_image.jpg \
    --confidence-threshold 0.5
```

#### Video Inference

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video path/to/football_clip.mp4 \
    --output output_video.mp4 \
    --confidence-threshold 0.5 \
    --skip-frames 1
```

**Inference Parameters:**
- `--confidence-threshold`: Minimum confidence score (default: 0.5)
- `--skip-frames`: Process every Nth frame (default: 1, process all frames)

#### Batch Inference on Frames

```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --frames-dir path/to/frames \
    --output path/to/output_frames \
    --confidence-threshold 0.5
```

### Programmatic Usage

```python
import torch
from src.models.detector import FootballDetector
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

# Access results
boxes = predictions['boxes']      # Bounding boxes [N, 4]
labels = predictions['labels']    # Class labels [N]
scores = predictions['scores']    # Confidence scores [N]
```

## Model Architecture

The model uses **Faster R-CNN** with the following components:

- **Backbone**: ResNet50 with Feature Pyramid Network (FPN)
- **Pre-training**: COCO dataset
- **Fine-tuning**: Transfer learning on football footage
- **Classes**: 3 classes (background, player, ball)

## Monitoring Training

Monitor training progress with TensorBoard:

```bash
tensorboard --logdir runs
```

This will display:
- Training and validation losses
- Loss components (classifier, box regression, RPN, etc.)
- Learning rate schedule

## Requirements

- Python 3.8+
- PyTorch 2.0+
- torchvision 0.15+
- OpenCV
- Albumentations (for data augmentation)
- See `requirements.txt` for complete list

## Data Preparation Tips

1. **Frame Extraction**: Extract frames from your football clips at a consistent frame rate
2. **Annotation**: Annotate frames with bounding boxes for players and the ball
3. **Format**: Ensure annotations follow the COCO format shown above
4. **Split**: Divide your data into training and validation sets (e.g., 80/20 split)
5. **Quality**: Higher quality annotations lead to better model performance

## Performance Optimization

- **GPU**: Use CUDA-enabled GPU for faster training and inference
- **Batch Size**: Adjust based on available GPU memory
- **Skip Frames**: For videos, process every Nth frame for faster inference
- **Backbone Layers**: Fine-tune number of trainable backbone layers

## Troubleshooting

**Out of Memory Errors:**
- Reduce batch size
- Reduce number of trainable backbone layers
- Use gradient accumulation

**Poor Detection Performance:**
- Increase training epochs
- Add more training data
- Adjust confidence threshold
- Check annotation quality

**Slow Inference:**
- Use GPU if available
- Increase skip_frames for video processing
- Reduce image resolution if acceptable

## Citation

This project is built for Deep Learning for Computer Vision using SoccerNet data.

## License

This project is available for educational and research purposes.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.
