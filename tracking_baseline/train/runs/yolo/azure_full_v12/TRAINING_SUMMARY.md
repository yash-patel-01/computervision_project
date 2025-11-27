# YOLOv8n Training Summary - Azure Full Dataset

## Training Configuration
- **Model**: YOLOv8n (3.0M parameters, 8.2 GFLOPs)
- **Dataset**: SoccerNet Tracking 2023
  - Train: 42,000 images
  - Val: 36,750 images
- **Classes**: 2 (player, ball)
- **Hardware**: Azure Tesla T4 GPU (16GB)
- **Training Time**: 10.15 hours (36,548 seconds)

## Hyperparameters
- Epochs: 50
- Batch size: 32
- Image size: 832x832
- Learning rate: 0.005 (auto-tuned)
- Optimizer: auto-selected
- Warmup epochs: 2
- Cache: RAM

## Augmentations
- Mosaic: 0.8 (disabled after epoch 40)
- Copy-paste: 0.2
- Scale: 0.7
- Translate: 0.1
- Horizontal flip: 0.5
- HSV adjustments: enabled

## Training Results (Validation Set)
| Metric | Value |
|--------|-------|
| mAP50 | 0.649 (64.9%) |
| mAP50-95 | 0.370 (37.0%) |
| Precision | 0.760 |
| Recall | 0.636 |
| Box Loss | 0.975 |
| Class Loss | 0.417 |

## Test Set Evaluation Results
Evaluated on full test set (36,750 images):

| Metric | Value |
|--------|-------|
| AP (IoU=0.50:0.95) | 0.339 (33.9%) |
| AP50 (IoU=0.50) | 0.599 (59.9%) |
| AP75 (IoU=0.75) | 0.351 (35.1%) |
| AP_small | 0.114 (11.4%) |
| AP_medium | 0.303 (30.3%) |
| AP_large | 0.674 (67.4%) |
| AR@100 | 0.396 (39.6%) |

## Training Progress
- **Epoch 1**: box_loss=1.484, cls_loss=1.004, mAP50=0.576
- **Epoch 10**: box_loss=1.211, cls_loss=0.556, mAP50=0.634
- **Epoch 20**: box_loss=1.147, cls_loss=0.513, mAP50=0.644
- **Epoch 30**: box_loss=1.103, cls_loss=0.487, mAP50=0.649
- **Epoch 40**: box_loss=1.066, cls_loss=0.466, mAP50=0.650
- **Epoch 50**: box_loss=0.975, cls_loss=0.417, mAP50=0.649

## Loss Reduction
- Box Loss: 34% reduction (1.484 → 0.975)
- Classification Loss: 58% reduction (1.004 → 0.417)

## Key Observations
1. Training converged smoothly over 50 epochs
2. Mosaic augmentation disabled at epoch 41 (close_mosaic=10) caused expected loss drop
3. Strong performance on large objects (AP_large=0.674)
4. Challenge with small objects (AP_small=0.114) - ball detection remains difficult
5. Validation metrics plateaued around epoch 30

## Files
- `weights/best.pt`: Best model weights (6.2MB)
- `weights/last.pt`: Final epoch weights
- `weights/epoch{N}.pt`: Checkpoints every 5 epochs
- `results.csv`: Per-epoch metrics
- `eval_metrics_yolo_full_test.json`: Full test evaluation metrics
- `args.yaml`: Complete training configuration

## Comparison to Subset Training
Previous subset training (750 images, 50 epochs):
- mAP50: 0.595
- Full dataset improvement: +5.4 percentage points

## Dataset Notes
- Used unique flattened filenames to prevent collisions across sequences
- Labels: YOLO format (class x_center y_center width height, normalized)
- Dataset location: `data/yolo_dataset_full_unique`
- Conversion script: `tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py`

## Reproducibility
To reproduce this training:
```bash
python tracking_baseline/train/train_yolo_ultra.py \
  --data data/yolo_dataset_full_unique/dataset.yaml \
  --epochs 50 \
  --batch 32 \
  --imgsz 832 \
  --cache ram \
  --project tracking_baseline/train/runs/yolo \
  --name azure_full_v12
```

## Date
Training completed: November 27, 2025
