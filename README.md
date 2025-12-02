# Football Video Analysis: Player Detection & Jersey Number Recognition

This project provides a complete pipeline for football video analysis using the SoccerNet dataset, combining two complementary computer vision tasks:

1. **Player & Ball Detection** - Detect and classify players and balls in football footage
2. **Jersey Number Recognition** - Identify and read jersey numbers from detected players

## Overview

### Part 1: Player & Ball Detection

The first component detects and classifies two object types in football footage:
- **Players** (class 0)
- **Ball** (class 1)

The original SoccerNet tracking dataset provides bounding boxes but doesn't distinguish between players and balls—everything shares one generic class. To address this, we implemented a heuristic-based algorithm that automatically identifies which tracks correspond to the ball based on size (small), shape (round), and consistency across frames. Note: multiple ball tracks can be identified in a sequence when the heuristic finds distinct small, round objects; the pipeline supports this.

**Models Trained:**
- Faster R-CNN (ResNet-50 FPN): AP = 8.4%
- YOLOv8n: **AP = 33.9%** ✓ (Recommended)

### Part 2: Jersey Number Recognition

The second component builds upon player detection to recognize jersey numbers:
- **Pose-based player extraction** using YOLOv8-Pose to identify back-facing players
- **Torso cropping** with quality filtering (shoulder width, visibility, sharpness)
- **Digit detection** using YOLOv8 trained on custom annotations
- **Number classification** using EfficientNet-B0 for individual digit recognition

This creates an end-to-end pipeline: detect players → extract jersey regions → recognize numbers.

## Datasets

### SoccerNet tracking-2023 (Player & Ball Detection)
- **Training set**: 58 sequences, ~42,000 annotated frames
- **Test set**: 49 sequences, ~36,750 annotated frames  
- **Challenge set**: Additional unannotated sequences

```python
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data")
downloader.password = "s0cc3rn3t"
downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
```

### SoccerNet jersey-2023 (Jersey Number Recognition)
- Video sequences with jersey number annotations
- Player identification with back-facing jersey visibility

```python
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="SoccerNetJersey")
downloader.password = "s0cc3rn3t"
downloader.downloadDataTask(task="jersey-2023", split=["train", "test", "challenge"])
```

## Complete Pipeline (From Scratch)

---

## PART 1: Player & Ball Detection Pipeline

Follow these steps in order to reproduce the player and ball detection project:

### Step 0: Setup Environment

```bash
# Install dependencies
pip install -r tracking_baseline/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1

# Download SoccerNet dataset (~190GB, takes several hours)
# See exploration.ipynb or use this Python snippet:
```
```python
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data")
downloader.password = "s0cc3rn3t"
downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
```

### Step 1: Generate Ball Labels

Process the tracking annotations to identify which tracks represent the ball using heuristics (size, shape, consistency):

```bash
# Quick sanity check (1 sequence per split)
python3 tracking_baseline/batch_ball_labeling.py --limit-seqs 1 --output-dir data/tmp_check

# Full run (all sequences)
python3 tracking_baseline/batch_ball_labeling.py
```

**Generates:**
- `data/ball_tracks.json` - Identified ball track IDs per sequence
- `data/coco_train.json` - Training annotations (~42K images)
- `data/coco_test.json` - Test annotations (~37K images)

### Step 2a: Train Faster R-CNN (Optional)

Train the two-stage Faster R-CNN detector:

```bash
cd tracking_baseline/train/runs/frcnn
python3 train_frcnn.py --epochs 6 --batch-size 4 --data-root ../../../../data
```

**Output:** Saves checkpoints and logs to `tracking_baseline/train/runs/frcnn/`

### Step 2b: Evaluate Faster R-CNN

```bash
# From tracking_baseline/train/runs/frcnn/
python3 eval_coco_frcnn.py --checkpoint model_epoch6.pth --data-root ../../../../data
```

**Output:** `eval_metrics_frcnn.json` with COCO metrics (AP, AP50, AP75, AP_small, etc.)

### Step 3a: Convert to YOLO Format

Convert COCO annotations to YOLO format with symlinked images:

```bash
cd tracking_baseline/train/runs/yolo
python3 convert_coco_to_yolo.py \
  --coco-train ../../../../data/coco_train.json \
  --coco-val ../../../../data/coco_test.json \
  --data-root ../../../../data \
  --out-root ../../../../data/yolo_dataset_full_unique \
  --workers 8 \
  --relative-path
```

**Generates:**
- `data/yolo_dataset_full_unique/labels/` - YOLO .txt files (~79K files)
- `data/yolo_dataset_full_unique/images/` - Symlinks to original images
- `data/yolo_dataset_full_unique/dataset.yaml` - Dataset configuration

### Step 3b: Train YOLOv8 (Recommended)

Train the single-stage YOLOv8 detector:

```bash
# From tracking_baseline/train/runs/yolo/
python3 train_yolo.py \
  --dataset-root ../../../../data/yolo_dataset_full_unique \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 16 \
  --imgsz 832
```

**Output:** Training results saved to `tracking_baseline/train/runs/yolo/<run_name>/`
- `weights/best.pt` - Best model checkpoint
- `results.csv` - Training metrics per epoch

**Note:** Pre-trained weights from our training are included at `azure_full_v12/weights/best.pt` (AP: 33.9%, AP50: 59.9%)

### Step 3c: Evaluate YOLOv8

```bash
# From tracking_baseline/train/runs/yolo/
python3 eval_coco_yolo.py \
  --weights azure_full_v12/weights/best.pt \
  --coco-test ../../../../data/coco_test.json \
  --data-root ../../../../data \
  --imgsz 832
```

**Output:** `eval_metrics_yolo.json` with COCO metrics

### Step 4: Compare Models

Review comprehensive evaluation and comparison in `model_evaluation.ipynb`:
- FRCNN performance analysis
- YOLOv8 performance analysis
- Side-by-side comparison (YOLOv8 outperforms by ~4x on AP)

---

## PART 2: Jersey Number Recognition Pipeline

Follow these steps to reproduce the jersey number recognition project:

### Prerequisites

Download the jersey-2023 dataset (see Datasets section above) and ensure it's in `SoccerNetJersey/jersey-2023/`.

### Step 1: Extract Player Torso Crops

Use pose estimation to identify and crop back-facing players from video frames:

**See:** `Jersey-Number_Recognition.ipynb` - Section: "Pick appropriate frames only"

This pipeline:
1. Loads video frames from jersey-2023 dataset
2. Applies YOLOv8-Pose to detect players and skeletal keypoints
3. Filters for back-facing players using geometric rules (shoulder visibility, face occlusion)
4. Extracts and crops torso regions with quality control (sharpness, aspect ratio)
5. Saves clean crops for annotation and training

### Step 2: Annotate Digits

Manually annotate digit bounding boxes in the cropped torso images using tools like [makesense.ai](https://www.makesense.ai/).

**Output:** YOLO-format annotation files (.txt) for each image

### Step 3: Train Digit Detector (YOLOv8)

Train YOLOv8 to detect individual digits (0-9) in jersey crops:

**See:** `Jersey-Number_Recognition.ipynb` - YOLO digit detection training section

- Model: YOLOv8n
- Classes: 10 (digits 0-9)
- Augmentations: Mosaic, copy-paste, HSV, rotation
- Output: Digit bounding boxes with class predictions

### Step 4: Train Digit Classifier (EfficientNet-B0)

Train a classification model for individual digit recognition:

**See:** `Jersey-Number_Recognition.ipynb` - EfficientNet training section

- Architecture: EfficientNet-B0 (pretrained on ImageNet)
- Input: Cropped digit regions from YOLO detector
- Output: 10-class softmax (digits 0-9)
- Training: Cross-entropy loss, data augmentation, learning rate scheduling

### Step 5: Full Pipeline Inference

Combine all components for end-to-end jersey number recognition:

1. Input: Video frame or image with players
2. Detect players (using Part 1 or pose model)
3. Extract back-facing player torsos
4. Detect digit regions (YOLOv8)
5. Classify each digit (EfficientNet-B0)
6. Reconstruct jersey number from left-to-right digit sequence

**See:** `Jersey-Number_Recognition.ipynb` - Inference and visualization sections

**Documentation:** `Jersey Number Recognition.docx` - Detailed methodology and results

## Project Structure

```
Code/
├── README.md                           # This file (joint project documentation)
├── .gitignore                          # Excludes large data files
│
├─── PART 1: Player & Ball Detection ───
├── exploration.ipynb                   # Data visualization and ball heuristic testing
├── model_evaluation.ipynb              # Model comparison (FRCNN vs YOLOv8)
├── model_comparison_demo.ipynb         # Side-by-side visual inference comparison
├── data/                               # Datasets (not in git)
│   ├── tracking-2023/                  # Player/ball tracking dataset
│   │   ├── train/                      # Training clips + annotations
│   │   ├── test/                       # Test clips + annotations
│   │   └── challenge2023/              # Challenge clips (no annotations)
│   ├── ball_tracks.json                # Generated: ball track IDs
│   ├── coco_train.json                 # Generated: training annotations (~42K images)
│   ├── coco_test.json                  # Generated: test annotations (~37K images)
│   └── yolo_dataset_full_unique/       # Generated: YOLO-format dataset
└── tracking_baseline/                  # Training and inference code
    ├── README.md                       # Detailed baseline documentation
    ├── requirements.txt                # Python dependencies
    ├── batch_ball_labeling.py          # Generate ball labels using heuristics
    └── train/
        ├── datasets/
        │   └── coco_ball.py            # Dataset loader
        └── runs/
            ├── frcnn/                  # Faster R-CNN scripts + outputs
            │   ├── train_frcnn.py      # Train Faster R-CNN
            │   ├── eval_coco_frcnn.py  # Evaluate FRCNN on COCO metrics
            │   ├── config_frcnn.json   # Training configuration
            │   ├── eval_metrics_frcnn.json # Test set metrics
            │   └── training_frcnn.log  # Training logs
            └── yolo/                   # YOLO scripts + outputs
                ├── train_yolo.py       # Train YOLOv8 on YOLO-format dataset
                ├── convert_coco_to_yolo.py # Convert COCO to YOLO format
                ├── eval_coco_yolo.py   # Evaluate YOLO checkpoint with COCO metrics
                └── azure_full_v12/     # Trained YOLOv8 results
                    └── weights/best.pt # Best model weights (AP: 33.9%)
│
├─── PART 2: Jersey Number Recognition ───
├── Jersey-Number_Recognition.ipynb     # Complete jersey number pipeline
├── Jersey Number Recognition.docx      # Detailed methodology documentation
└── SoccerNetJersey/                    # Jersey recognition workspace (not in git)
    ├── jersey-2023/                    # Jersey dataset
    │   ├── train/                      # Training data
    │   │   ├── images/                 # Original video frames
    │   │   ├── train_gt.json           # Ground truth annotations
    │   │   └── images_back_crops_*/    # Extracted player crops
    │   └── test/                       # Test data (similar structure)
    ├── our_digits_yolo_crop/           # Digit detection dataset
    │   ├── labels/                     # YOLO annotations for digits
    │   ├── images/                     # Annotated digit images
    │   └── digit.yaml                  # YOLO dataset config
    ├── checkpoints_EffNetB0_capp300/   # EfficientNet model checkpoints
    │   └── EffNetB0_capp300_best.pt    # Best digit classifier
    └── yolov8n-pose.pt                 # Pose estimation model weights

```

## Quick Start

### Player & Ball Detection (Pre-trained Models)

If you want to skip training and use our pre-trained models:

1. Download SoccerNet tracking-2023 dataset
2. Generate ball labels: `python3 tracking_baseline/batch_ball_labeling.py`
3. Use pre-trained YOLOv8 weights: `tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt`
4. Run evaluation or inference with `eval_coco_yolo.py`

### Jersey Number Recognition (Pre-trained Models)

Use the trained models from `SoccerNetJersey/`:
1. Download SoccerNet jersey-2023 dataset
2. Load digit detector: YOLOv8 trained on digit annotations
3. Load digit classifier: `SoccerNetJersey/checkpoints_EffNetB0_capp300/EffNetB0_capp300_best.pt`
4. Run inference pipeline in `Jersey-Number_Recognition.ipynb`

## Key Notebooks

- **`exploration.ipynb`** - Data exploration, ball heuristic visualization (Part 1)
- **`model_evaluation.ipynb`** - Comprehensive model comparison: FRCNN vs YOLOv8 (Part 1)
- **`model_comparison_demo.ipynb`** - Side-by-side visual inference comparison (Part 1)
- **`Jersey-Number_Recognition.ipynb`** - Complete jersey number pipeline (Part 2)

## Documentation

- **`README.md`** (this file) - Joint project overview and setup
- **`tracking_baseline/README.md`** - Detailed player/ball detection documentation
- **`Jersey Number Recognition.docx`** - Jersey number recognition methodology and results
- **`.gitignore`** - Excludes large files (datasets, checkpoints) from version control

## Requirements

### Common Requirements
- Python 3.10+
- PyTorch 2.5+ with CUDA support (GPU strongly recommended)
- OpenCV, NumPy, Matplotlib
- SoccerNet API (`pip install SoccerNet`)

### Part 1: Player & Ball Detection
- torchvision 0.20+
- pycocotools (for COCO evaluation)
- ultralytics (for YOLOv8)
- ~190GB disk space for tracking-2023 dataset

See: `tracking_baseline/requirements.txt`

### Part 2: Jersey Number Recognition
- ultralytics (YOLOv8 and YOLOv8-Pose)
- torchvision (EfficientNet-B0)
- PIL/Pillow (image processing)
- Additional disk space for jersey-2023 dataset and crops

See: `Jersey-Number_Recognition.ipynb` for detailed setup

## Results Summary

### Player & Ball Detection
| Model | AP | AP50 | AP75 | AP_small (Ball) | AP_large (Players) |
|-------|-----|------|------|-----------------|-------------------|
| Faster R-CNN | 8.4% | 22.6% | 7.7% | 8.5% | 0.0% |
| **YOLOv8n** | **33.9%** | **59.9%** | **31.2%** | **11.4%** | **67.4%** |

**Winner:** YOLOv8n - 4x better overall performance, excellent player detection

### Jersey Number Recognition
- **Pose-based extraction:** Successfully isolates back-facing players
- **Digit detection (YOLOv8):** Accurately localizes individual digits on jerseys
- **Digit classification (EfficientNet-B0):** High accuracy on digit recognition
- **End-to-end pipeline:** Combines detection → extraction → recognition for complete jersey number identification

See `Jersey Number Recognition.docx` for detailed results and methodology.

## Citation

If you use this work, please cite the SoccerNet dataset:

```
@inproceedings{Deliege2021SoccerNetv2,
  title={SoccerNet-v2: A Dataset and Benchmarks for Holistic Understanding of Broadcast Soccer Videos},
  author={Deliège, Adrien and others},
  booktitle={CVPR Workshops},
  year={2021}
}
```
