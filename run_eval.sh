#!/bin/bash
# Evaluation script for Azure server
# Run from: /Code directory

set -e  # Exit on error

echo "=== Evaluating Faster R-CNN on Test Set ==="
echo "Working directory: $(pwd)"

# Navigate to training directory
cd tracking_baseline/train

# Run evaluation on test set
echo "Evaluating on coco_test.json (36,750 images)..."
python eval_coco.py \
  --checkpoint runs/frcnn/model_epoch6.pth \
  --coco-json ../../data/coco_test.json \
  --data-root ../../data

echo ""
echo "=== Evaluation Complete ==="
echo "Metrics saved to: tracking_baseline/train/runs/frcnn/eval_metrics.json"
