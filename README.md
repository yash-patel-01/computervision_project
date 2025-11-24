# Football Player & Ball Detection Project

Deep Learning for Computer Vision project: Train a Faster R-CNN model to detect players and balls in football video clips.

## Project Goal

Detect and classify objects in football clips:
- **Players** (class 0)
- **Ball** (class 1)

**Challenge:** Original dataset labels all objects with the same class. We use a heuristic algorithm to automatically identify ball tracks based on size, shape, and movement patterns.

## Dataset

SoccerNet tracking-2023 dataset:
- **train/**: 58 sequences, 42,000 images with bounding boxes
- **test/**: 49 sequences, 36,750 images with bounding boxes  
- **challenge2023/**: Unannotated clips for inference

Download via SoccerNet Downloader:
```python
from SoccerNet.Downloader import SoccerNetDownloader
downloader = SoccerNetDownloader(LocalDirectory="data")
downloader.password = "s0cc3rn3t"
downloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
```

## Workflow

### Step 1: Generate Ball Labels

```bash
python3 tracking_baseline/src/inference/batch_ball_labeling.py
```

Identifies ball tracks using heuristics (size, roundness, presence). Generates:
- `data/ball_tracks.json`: Ball track IDs per sequence
- `data/coco_train.json`: Training annotations (42K images, 103MB)
- `data/coco_test.json`: Test annotations (36K images, 82MB)

Manual overrides: `tracking_baseline/data/overrides.json`

### Step 2: Train Model

```bash
cd tracking_baseline/train
python3 train_frcnn.py --epochs 6 --batch-size 4 --data-root ../../data
```

Trains Faster R-CNN on 42K training images. Outputs checkpoints to `runs/frcnn/model_epoch{N}.pth`. 

Options: `--epochs`, `--batch-size`, `--lr`, `--coco-json`, `--output-dir`

### Step 3: Evaluate

```bash
cd tracking_baseline/train
python3 eval_coco.py --checkpoint runs/frcnn/model_epoch6.pth --data-root ../../data
```

Computes COCO metrics (AP, AP50, AP75, AP_small) on 36K test images. Results saved to `runs/frcnn/eval_metrics.json`.

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
    ├── configs/
    │   └── tracker.yaml                # Tracker configuration
    ├── data/
    │   └── overrides.json              # Manual corrections for ball labeling
    ├── src/
    │   ├── inference/
    │   │   ├── batch_ball_labeling.py  # Generate ball labels (Step 1)
    │   │   ├── run_tracking.py         # Run tracker on detections
    │   │   └── export_per_sequence.py  # Export per-sequence results
    │   ├── tracking/                   # Tracking algorithms (Kalman, Hungarian)
    │   └── utils/                      # Helper functions (IoU, box utils)
    └── train/
        ├── train_frcnn.py              # Train Faster R-CNN (Step 2)
        ├── eval_coco.py                # Evaluate model on COCO metrics
        └── datasets/
            └── coco_ball.py            # Dataset loader

```

## Notes

- `exploration.ipynb`: Data visualization and ball identification examples
- See `tracking_baseline/README.md` for tracking algorithm details
- Generated files excluded from git: datasets, annotations, checkpoints, training outputs

## Installation

```bash
pip install -r tracking_baseline/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121  # For GPU
```

## Requirements

- Python 3.10+, PyTorch 2.5+, CUDA support recommended
- ~190GB disk space for dataset
