# Football Player & Ball Detection — Baseline

This baseline covers detection and data preparation. Tracking modules referenced earlier are not included in this repository snapshot.

Pipeline
- Heuristic ball identification from MOT GT (supports multiple ball tracks per sequence).
- COCO-style dataset generation for train/test with categories: `player` (0), `ball` (1).
- Faster R-CNN training on generated COCO; COCO evaluation of checkpoints.

What's included
- `batch_ball_labeling.py` — Generate `ball_tracks.json`, `coco_train.json`, `coco_test.json`.
- `train/datasets/coco_ball.py` — Minimal COCO loader for torchvision detection models.
- `train/runs/frcnn/train_frcnn.py` — Train Faster R-CNN (2 classes + background).
- `train/runs/frcnn/eval_coco_frcnn.py` — Evaluate FRCNN checkpoints with `pycocotools` (AP, AP50, AP75, AP_small, …).
- `train/runs/yolo/convert_coco_to_yolo.py` — Convert COCO annotations to YOLO format with symlinked images.
- `train/runs/yolo/train_yolo.py` — Train YOLOv8 using Ultralytics native training on YOLO dataset.
- `train/runs/yolo/eval_coco_yolo.py` — Evaluate YOLOv8 checkpoints with COCO metrics.

Quick start
```bash
# 1. Generate ball labels (from project root)
python3 tracking_baseline/batch_ball_labeling.py --limit-seqs 1 --output-dir data/tmp_check
python3 tracking_baseline/batch_ball_labeling.py  # full run

# 2. Train FRCNN
cd tracking_baseline/train/runs/frcnn
python3 train_frcnn.py --epochs 6 --batch-size 4 --data-root ../../../../data

# 3. Evaluate FRCNN
python3 eval_coco_frcnn.py --checkpoint model_epoch6.pth --data-root ../../../../data

# 4. Convert COCO to YOLO format
cd ../yolo
python3 convert_coco_to_yolo.py \
  --coco-train ../../../../data/coco_train.json \
  --coco-val ../../../../data/coco_test.json \
  --data-root ../../../../data \
  --out-root ../../../../data/yolo_dataset_full_unique \
  --workers 8 --relative-path

# 5. Train YOLOv8
python3 train_yolo.py \
  --dataset-root ../../../../data/yolo_dataset_full_unique \
  --model yolov8n.pt --epochs 50 --batch 16 --imgsz 832
```

Notes
- Manual overrides are not expected; the heuristic automatically identifies ball tracks. Adjust outputs directly if needed.
- Pre-trained YOLOv8 model available at `train/runs/yolo/azure_full_v12/weights/best.pt` (AP: 33.9%, AP50: 59.9%)
- See `model_evaluation.ipynb` for comprehensive performance comparison between FRCNN and YOLOv8

Potential improvements
- Optional CLI to cap or adjust the maximum number of ball tracks per sequence
- Per-class instance balancing during training if ball frequency is low
- Fine-tune detection thresholds for specific use cases (precision vs recall trade-off)

