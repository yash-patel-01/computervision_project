# Football Player & Ball Detection Project

Deep Learning for Computer Vision course project: Detecting players and the ball in short football video clips.

## Project Goal

Train an object detection model (Faster R-CNN) to identify:
- **Players** (class 0)
- **Ball** (class 1)

The model outputs bounding box coordinates for each detected object in video frames, similar to the training data format.

## Dataset

- **Training data:** `data/tracking-2023/train/` and `data/tracking-2023/test/`
  - Contains annotated clips with ground truth bounding boxes
  - Ball labels are not provided - we generate them automatically
  
- **Challenge data:** `data/tracking-2023/challenge2023/`
  - Unannotated clips where we apply our trained model

**Note:** Data directories are excluded from git (see `.gitignore`). Download the data yourself:
```python
from SoccerNet.Downloader import SoccerNetDownloader
mySoccerNetDownloader = SoccerNetDownloader(LocalDirectory="data")
mySoccerNetDownloader.password = "s0cc3rn3t"
mySoccerNetDownloader.downloadDataTask(task="tracking-2023", split=["train", "test", "challenge"])
```

## Workflow

### Step 1: Generate Ball Labels

The training data doesn't label which objects are balls, so we use a heuristic algorithm to identify them automatically:

```bash
cd tracking_baseline/src/inference
python batch_ball_labeling.py
```

**What it does:**
- Analyzes all tracks across train/test sequences
- Identifies balls using size, roundness, and movement patterns
- Creates `data/ball_tracks.json` (ball track IDs per sequence)
- Creates `data/coco_pseudo.json` (merged COCO-format annotations for all sequences)

**Note:** `coco_pseudo.json` is large (182MB) and excluded from git.

### Step 2: Train the Detection Model

Train Faster R-CNN on the labeled data:

```bash
cd tracking_baseline/train
python train_frcnn.py --epochs 10 --batch-size 4
```

**Options:**
- `--epochs`: Number of training epochs (default: 6)
- `--batch-size`: Batch size (default: 4)
- `--lr`: Learning rate (default: 0.005)
- `--coco-json`: Path to annotations (default: `data/coco_pseudo.json`)
- `--output-dir`: Where to save model checkpoints (default: `train/runs/frcnn/`)

**Output:**
- Model checkpoints saved to `tracking_baseline/train/runs/frcnn/`
- `model_best.pth`: Best model based on validation loss
- `model_epoch{N}.pth`: Checkpoint after each epoch

### Step 3: Run Inference (Coming Soon)

Apply your trained model to generate predictions on the challenge clips.

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
│   └── coco_pseudo.json                # Generated: merged training annotations (182MB)
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

## Files in This Repo

**Included in git:**
- Python training/inference code
- Notebook for exploration
- Configuration files
- Documentation

**Excluded from git:**
- Large datasets (`tracking-2023/`, `jersey-2023/`, `per_seq/`)
- `coco_pseudo.json` (182MB)
- Model checkpoints (`.pth`, `.pt`, `.h5`)
- Training outputs (`train/runs/`)

## Getting Started

1. **Clone this repo**
2. **Download the data** (see Dataset section)
3. **Install dependencies:** `pip install -r tracking_baseline/requirements.txt`
4. **Follow the workflow** above (Steps 1-3)

## Notes

- The `exploration.ipynb` notebook shows data visualization and the ball identification algorithm in action
- You can manually override ball labels by editing `tracking_baseline/data/overrides.json`
- For detailed model architecture and tracking algorithm info, see `tracking_baseline/README.md`
