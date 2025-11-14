# System Architecture

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Football Tracking System                     │
└─────────────────────────────────────────────────────────────────┘

┌──────────────────┐         ┌──────────────────┐
│  Training Phase  │         │  Inference Phase │
└──────────────────┘         └──────────────────┘
         │                            │
         v                            v
┌─────────────────┐         ┌─────────────────┐
│ Football Clips  │         │  New Footage    │
│   (Frames)      │         │  (Image/Video)  │
└─────────────────┘         └─────────────────┘
         │                            │
         v                            v
┌─────────────────┐         ┌─────────────────┐
│  Annotations    │         │  Pre-process    │
│  (COCO JSON)    │         │   & Transform   │
└─────────────────┘         └─────────────────┘
         │                            │
         v                            v
┌─────────────────────────────────────────────┐
│           Faster R-CNN Model                │
│  ┌──────────────────────────────────────┐  │
│  │  ResNet50 + FPN Backbone             │  │
│  │  (Pre-trained on COCO)               │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  Region Proposal Network (RPN)       │  │
│  └──────────────────────────────────────┘  │
│  ┌──────────────────────────────────────┐  │
│  │  ROI Pooling & Classification Head   │  │
│  └──────────────────────────────────────┘  │
└─────────────────────────────────────────────┘
         │                            │
         v                            v
┌─────────────────┐         ┌─────────────────┐
│  Save Model     │         │  Predictions    │
│  Checkpoint     │         │  (Boxes+Labels) │
└─────────────────┘         └─────────────────┘
                                     │
                                     v
                            ┌─────────────────┐
                            │  Visualization  │
                            │  (Colored Boxes)│
                            └─────────────────┘
```

## Data Flow

### Training Data Flow
```
Raw Video → Frame Extraction → Annotation Tool → COCO JSON
                                                      ↓
                                              FootballTrackingDataset
                                                      ↓
                                              Data Augmentation
                                                      ↓
                                              DataLoader (Batches)
                                                      ↓
                                              Faster R-CNN Model
                                                      ↓
                                              Loss Calculation
                                                      ↓
                                              Backpropagation
                                                      ↓
                                              Checkpoint Saving
```

### Inference Data Flow
```
Input (Image/Video) → Pre-processing → Normalization → Model Forward Pass
                                                              ↓
                                                    Bounding Box Predictions
                                                              ↓
                                                    Confidence Filtering
                                                              ↓
                                                    Visualization (Draw Boxes)
                                                              ↓
                                                    Output (Annotated Media)
```

## Component Architecture

### 1. Data Layer (`src/data/`)
```
FootballTrackingDataset
├── __init__(root_dir, annotation_file, transforms)
├── __len__() → int
├── __getitem__(idx) → (image, target)
└── Loads and parses COCO JSON annotations

Transforms
├── get_train_transforms() → Compose
│   ├── HorizontalFlip
│   ├── RandomBrightnessContrast
│   ├── Blur
│   └── Normalize
└── get_val_transforms() → Compose
    └── Normalize
```

### 2. Model Layer (`src/models/`)
```
FootballDetector
├── __init__(num_classes, device)
├── train_mode()
├── eval_mode()
├── forward(images, targets) → losses/predictions
├── predict(images, threshold) → filtered_predictions
├── save(path)
└── load(path)

create_model(num_classes, pretrained)
└── Returns Faster R-CNN with custom head
```

### 3. Utility Layer (`src/utils/`)
```
Visualization Module
├── draw_bounding_boxes(image, boxes, labels, scores)
├── visualize_predictions(image, predictions)
├── visualize_batch(images, targets, predictions)
└── save_video_with_predictions(video_path, predictions, output_path)

Color Mapping
├── Player → Green (0, 255, 0)
└── Ball → Red/Blue (255, 0, 0 in BGR)
```

### 4. Training Pipeline (`src/train.py`)
```
Training Loop
├── Load datasets (train + val)
├── Create data loaders
├── Initialize model
├── Setup optimizer (SGD + momentum)
├── Setup scheduler (StepLR)
├── For each epoch:
│   ├── train_one_epoch()
│   │   ├── Forward pass
│   │   ├── Calculate losses
│   │   ├── Backward pass
│   │   └── Update weights
│   ├── evaluate()
│   │   └── Validation loss
│   ├── Update learning rate
│   └── Save checkpoint
└── Save best/final models
```

### 5. Inference Pipeline (`src/inference.py`)
```
Inference Types
├── predict_image(model, image_path, output_path)
│   └── Single image detection
├── predict_video(model, video_path, output_path)
│   ├── Read frame
│   ├── Run detection
│   ├── Draw boxes
│   └── Write frame
└── predict_frames(model, frames_dir, output_dir)
    └── Batch process directory
```

## Model Architecture Details

### Faster R-CNN Components
```
Input Image (H x W x 3)
        ↓
┌───────────────────────────────┐
│   Backbone: ResNet50 + FPN    │
│   - Extract multi-scale       │
│     feature maps               │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│  Region Proposal Network      │
│  - Generates ~2000 proposals  │
│  - Objectness scores          │
│  - Bounding box refinement    │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│     ROI Align & Pooling       │
│  - Extract fixed-size         │
│    features for each proposal │
└───────────────────────────────┘
        ↓
┌───────────────────────────────┐
│   Classification Head         │
│  - Class prediction (3)       │
│  - Box regression refinement  │
└───────────────────────────────┘
        ↓
Output: Boxes [N x 4], Labels [N], Scores [N]
```

### Loss Components
```
Total Loss = λ₁ * L_cls + λ₂ * L_box + λ₃ * L_rpn_cls + λ₄ * L_rpn_box

Where:
- L_cls:     Classification loss (CrossEntropy)
- L_box:     Bounding box regression loss (Smooth L1)
- L_rpn_cls: RPN objectness loss
- L_rpn_box: RPN box regression loss
```

## Configuration Flow
```
configs/default_config.yaml
        ↓
    YAML Parser
        ↓
┌─────────────────────┐
│  Configuration Dict │
├─────────────────────┤
│ - model params      │
│ - training params   │
│ - data paths        │
│ - inference params  │
└─────────────────────┘
        ↓
  Applied to Scripts
```

## Execution Workflow

### Training Workflow
```
1. User prepares data
   └── Images + COCO JSON annotations

2. Run training script
   └── python src/train.py --config configs/default_config.yaml

3. Training process
   ├── Epoch 1
   │   ├── Train on training set
   │   ├── Validate on validation set
   │   └── Save checkpoint if best
   ├── Epoch 2
   │   └── ...
   └── Epoch N
       └── Save final checkpoint

4. Monitor with TensorBoard
   └── tensorboard --logdir runs

5. Output: checkpoints/best_model.pth
```

### Inference Workflow
```
1. User has trained model
   └── checkpoints/best_model.pth

2. Run inference script
   └── python src/inference.py --checkpoint ... --video input.mp4 --output output.mp4

3. Inference process
   ├── Load model
   ├── For each frame:
   │   ├── Pre-process
   │   ├── Run detection
   │   ├── Filter by confidence
   │   └── Draw bounding boxes
   └── Save output video

4. Output: output.mp4 with annotated boxes
```

## Class Hierarchy

```
torch.utils.data.Dataset
        ↑
        │
FootballTrackingDataset

torch.nn.Module
        ↑
        │
FasterRCNN (from torchvision)
        ↑
        │
FootballDetector (wrapper)
```

## File Organization by Purpose

### Core Functionality
- `src/models/detector.py` - Model definition
- `src/data/dataset.py` - Data loading
- `src/train.py` - Training logic
- `src/inference.py` - Inference logic

### Support Utilities
- `src/utils/visualization.py` - Visualization tools

### Configuration
- `configs/default_config.yaml` - Default settings

### Documentation
- `README.md` - Main documentation
- `QUICK_START.md` - Getting started guide
- `ARCHITECTURE.md` - This file
- `IMPLEMENTATION_SUMMARY.md` - Implementation details

### Examples & Tests
- `example_usage.py` - Usage examples
- `tests/test_structure.py` - Structure validation

### Setup
- `requirements.txt` - Dependencies
- `setup.py` - Package installation
- `.gitignore` - Git exclusions

## Technology Stack

```
┌─────────────────────────────────────┐
│         Application Layer           │
│  Training Script | Inference Script │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│          Framework Layer            │
│      PyTorch | Torchvision          │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│         Processing Layer            │
│  OpenCV | Albumentations | NumPy    │
└─────────────────────────────────────┘
              ↓
┌─────────────────────────────────────┐
│          System Layer               │
│     CUDA | cuDNN (optional)         │
└─────────────────────────────────────┘
```

## Extensibility Points

The architecture is designed to be extensible:

1. **New Model Architectures**
   - Modify `create_model()` in `src/models/detector.py`
   - Can swap Faster R-CNN with YOLO, RetinaNet, etc.

2. **Additional Classes**
   - Update `num_classes` parameter
   - Add categories to annotations
   - Update color mapping in visualization

3. **Custom Data Augmentation**
   - Modify transforms in `src/data/dataset.py`
   - Add new Albumentations transforms

4. **Different Data Formats**
   - Extend `FootballTrackingDataset` class
   - Implement custom `__getitem__` method

5. **Post-Processing**
   - Add tracking algorithms (DeepSORT, etc.)
   - Implement in inference pipeline

6. **Evaluation Metrics**
   - Add metric computation in training loop
   - Implement custom evaluation functions
