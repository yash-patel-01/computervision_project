# Football Player & Ball Detection — Baseline

This baseline covers detection and data preparation. Tracking modules referenced earlier are not included in this repository snapshot.

Pipeline
- Heuristic ball identification from MOT GT (supports multiple ball tracks per sequence).
- COCO-style dataset generation for train/test with categories: `player` (0), `ball` (1).
- Faster R-CNN training on generated COCO; COCO evaluation of checkpoints.

What’s included
- `src/inference/batch_ball_labeling.py` — Generate `ball_tracks.json`, `coco_train.json`, `coco_test.json`.
- `train/datasets/coco_ball.py` — Minimal COCO loader for torchvision detection models.
- `train/train_frcnn.py` — Train Faster R-CNN (2 classes + background).
- `train/eval_coco.py` — Evaluate FRCNN checkpoints with `pycocotools` (AP, AP50, AP75, AP_small, …).
- `train/runs/yolo/convert_coco_to_yolo.py` — Convert COCO annotations to YOLO format with symlinked images.
- `train/runs/yolo/train_yolo_ultra.py` — Train YOLOv8 using Ultralytics native training on YOLO dataset.
- `train/runs/yolo/eval_coco_yolo.py` — Evaluate YOLOv8 checkpoints with COCO metrics.

Quick start
```bash
# From project root
python3 tracking_baseline/src/inference/batch_ball_labeling.py --limit-seqs 1 --output-dir data/tmp_check
python3 tracking_baseline/src/inference/batch_ball_labeling.py  # full run

# FRCNN (under tracking_baseline/train)
python3 tracking_baseline/train/train_frcnn.py --dry-run --limit 8
# YOLO (under tracking_baseline/train/runs/yolo)
# First, generate YOLO dataset from COCO
python3 tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py \
  --data-root data --out-root data/yolo_dataset_full_abs \
  --workers 8 --assume-width 1920 --assume-height 1080

# Then train YOLOv8
python3 tracking_baseline/train/runs/yolo/train_yolo_ultra.py \
  --dataset-root data/yolo_dataset_full_abs --model yolov8n.pt \
  --epochs 50 --batch 16 --imgsz 832 --lr0 0.005
```

Notes
- Manual overrides are not expected anymore; adjust outputs directly if needed.
- Tracking code (`src/tracking`, `src/utils`, configs) is currently a placeholder. If you plan tracking-by-detection, add those modules or remove references.

Potential improvements
- Optional CLI to cap or adjust the maximum number of ball tracks per sequence.
- Per-class instance balancing during training if ball frequency is low.

