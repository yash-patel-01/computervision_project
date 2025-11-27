# Football Player & Ball Detection

This project trains a Faster R-CNN model to detect players and balls in football video sequences from the SoccerNet dataset.

## Overview

The main objective is detecting and classifying two object types in football footage:
- **Players** (class 0)
- **Ball** (class 1)

The original SoccerNet tracking dataset provides bounding boxes but doesn't distinguish between players and balls—everything shares one generic class. To address this, I implemented a heuristic-based algorithm that automatically identifies which tracks correspond to the ball based on size (small), shape (round), and consistency across frames. Note: multiple ball tracks can be identified in a sequence when the heuristic finds distinct small, round objects; the pipeline supports this.

## Dataset

Using SoccerNet tracking-2023:
- **Training set**: 58 sequences, ~42,000 annotated frames
- **Test set**: 49 sequences, ~36,750 annotated frames  
- **Challenge set**: Additional unannotated sequences

The dataset can be downloaded using the SoccerNet API:
```python
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data")
downloader.password = "s0cc3rn3t"
downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
```

## Complete Pipeline (From Scratch)

Follow these steps in order to reproduce the entire project:

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

## Project Structure

```
Code/
├── README.md                           # This file
├── exploration.ipynb                   # Data visualization and testing
├── .gitignore                          # Excludes large data files
├── data/                               # Datasets (not in git)
│   ├── tracking-2023/                  # Main dataset
│   │   ├── train/                      # Training clips + annotations
│   │   ├── test/                       # Test clips + annotations
│   │   └── challenge2023/              # Challenge clips (no annotations)
│   ├── ball_tracks.json                # Generated: ball track IDs
│   ├── coco_train.json                 # Generated: training annotations (~42K images)
│   └── coco_test.json                  # Generated: test annotations (~37K images)
└── tracking_baseline/                  # Training and inference code
    ├── README.md                       # Detailed baseline documentation
    ├── requirements.txt                # Python dependencies
    ├── batch_ball_labeling.py          # Generate ball labels (Step 1)
    └── train/
        ├── datasets/
        │   └── coco_ball.py            # Dataset loader
        └── runs/
            ├── frcnn/                  # Faster R-CNN scripts + outputs
            │   ├── train_frcnn.py      # Train Faster R-CNN (Step 2)
            │   ├── eval_coco_frcnn.py  # Evaluate FRCNN on COCO metrics
            │   ├── config_frcnn.json   # Training configuration
            │   ├── eval_metrics_frcnn.json # Test set metrics
            │   └── training_frcnn.log  # Training logs
            └── yolo/                   # YOLO scripts + outputs
                ├── train_yolo.py       # Train YOLOv8 on YOLO-format dataset
                ├── convert_coco_to_yolo.py # Convert COCO to YOLO format
                ├── eval_coco_yolo.py   # Evaluate YOLO checkpoint with COCO metrics
                └── azure_full_v12/     # Trained YOLOv8 results
                    └── weights/best.pt # Best model weights

```

## Quick Start (Using Pre-trained Models)

If you want to skip training and use our pre-trained models:

1. Download the SoccerNet dataset (Step 0 above)
2. Generate ball labels (Step 1 above)
3. Use pre-trained YOLOv8 weights: `tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt`
4. Run evaluation or inference directly

## Additional Files

- `exploration.ipynb` - Data exploration and visualization notebook
- `model_evaluation.ipynb` - Comprehensive model comparison and results
- `tracking_baseline/README.md` - Detailed baseline documentation
- `.gitignore` - Excludes large files (datasets, checkpoints) from version control

## Requirements

- Python 3.10+
- PyTorch 2.5+ with CUDA support
- ~190GB disk space for full dataset
- GPU strongly recommended for training (CPU inference is feasible)
