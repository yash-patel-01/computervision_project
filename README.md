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

## Pipeline

### 1. Ball Labeling

The first step processes the tracking annotations to identify which tracks represent the ball:

```bash
# Quick sanity run (limit sequences per split, choose output dir)
python3 tracking_baseline/src/inference/batch_ball_labeling.py --limit-seqs 1 --output-dir data/tmp_check

# Full run
python3 tracking_baseline/src/inference/batch_ball_labeling.py
```

This generates COCO-format annotations with proper class labels:
- `data/ball_tracks.json` - Identified ball track IDs per sequence
- `data/coco_train.json` - Training annotations (~42K images)
- `data/coco_test.json` - Test annotations (~37K images)

The heuristic looks for small, round, consistently-present objects. Manual overrides are no longer required or expected; if you need to adjust results, edit the generated JSONs.

### 2. Model Training

Train the Faster R-CNN detector on the labeled data:

```bash
cd tracking_baseline/train
python3 train_frcnn.py --epochs 6 --batch-size 4 --data-root ../../data
```

Training takes several hours on GPU and saves checkpoints to `runs/frcnn/`. Key parameters can be adjusted via command-line flags (`--epochs`, `--batch-size`, `--lr`).

### 3. Evaluation

Measure detection performance on the test set:

```bash
python3 eval_coco.py --checkpoint runs/frcnn/model_epoch6.pth --data-root ../../data --progress-interval 500
```

Adjust `--progress-interval` (images per progress print) or omit it entirely to disable progress output.

This computes standard COCO metrics (AP, AP50, AP75, etc.) and saves results to `runs/frcnn/eval_metrics.json`. The AP_small metric is particularly important since it specifically measures ball detection accuracy.

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
    ├── configs/                        # (Reserved for tracker configs; not currently used)
    ├── src/
    │   ├── inference/
    │   │   ├── batch_ball_labeling.py  # Generate ball labels (Step 1)
    │   ├── tracking/                   # (Placeholder; tracking modules not committed)
    │   └── utils/                      # (Placeholder; helper modules not committed)
    └── train/
        ├── train_frcnn.py              # Train Faster R-CNN (Step 2)
        ├── eval_coco.py                # Evaluate FRCNN on COCO metrics
        ├── runs/
        │   ├── frcnn/                  # Faster R-CNN run outputs
        │   └── yolo/                   # YOLO scripts + run outputs for symmetry
        │       ├── train_yolo_ultra.py # Train YOLOv8 on YOLO-format dataset
        │       ├── convert_coco_to_yolo.py # Convert COCO to YOLO format
        │       └── eval_coco_yolo.py   # Evaluate YOLO checkpoint with COCO metrics
        └── datasets/
            └── coco_ball.py            # Dataset loader

```

## Additional Files

- `exploration.ipynb` - Data exploration and visualization notebook
- `tracking_baseline/README.md` - More detailed documentation
- `.gitignore` - Excludes large files (datasets, checkpoints) from version control

## Setup

Install dependencies:
```bash
pip install -r tracking_baseline/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # CUDA 12.1
```

Requirements: Python 3.10+, PyTorch 2.5+, ~190GB disk space for the full dataset. GPU strongly recommended for training.
