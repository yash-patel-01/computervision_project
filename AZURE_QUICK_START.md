# Azure Training Quick Reference

## Before Starting
1. Ensure data is uploaded to Azure:
   - `data/tracking-2023/` (train & test folders)
   - `data/coco_train.json` (42K images)
   - `data/coco_test.json` (37K images)
   - `data/ball_tracks.json`

2. If data not present, generate labels:
   ```bash
   cd /path/to/Code
   python tracking_baseline/src/inference/batch_ball_labeling.py
   ```

## Training Commands

### Quick Start (using scripts)
```bash
cd /path/to/Code
./run_training.sh    # Train model
./run_eval.sh        # Evaluate on test set
```

### Manual Commands

**Train:**
```bash
cd tracking_baseline/train
python train_frcnn.py \
  --epochs 6 \
  --batch-size 4 \
  --lr 0.005 \
  --coco-json ../../data/coco_train.json \
  --data-root ../../data \
  --output-dir runs/frcnn
```

**Evaluate:**
```bash
python eval_coco.py \
  --checkpoint runs/frcnn/model_epoch6.pth \
  --coco-json ../../data/coco_test.json \
  --data-root ../../data
```

## Monitoring

**Check GPU usage:**
```bash
watch -n 1 nvidia-smi
```

**View training in real-time (if using tmux):**
```bash
tmux attach -t training
# Detach: Ctrl+B then D
```

## Expected Output

- Training: ~2-3 hours on V100/T4 GPU
- Checkpoints: `tracking_baseline/train/runs/frcnn/model_epoch{N}.pth`
- Metrics: `tracking_baseline/train/runs/frcnn/eval_metrics.json`

## Key Files Summary
- **Train data:** 42,000 images from 58 sequences
- **Test data:** 36,750 images from 49 sequences
- **Classes:** 0=player, 1=ball
- **Model:** Faster R-CNN with ResNet50-FPN backbone
