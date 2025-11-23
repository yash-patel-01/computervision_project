#!/bin/bash
# Training script for Azure server
# Run from: /Code directory

set -e  # Exit on error

echo "=== Starting Faster R-CNN Training ==="
echo "Working directory: $(pwd)"

# Navigate to training directory
cd tracking_baseline/train

# Run training on train set
echo "Training on coco_train.json (42,000 images)..."
python train_frcnn.py \
  --epochs 6 \
  --batch-size 4 \
  --lr 0.005 \
  --coco-json ../../data/coco_train.json \
  --data-root ../../data \
  --output-dir runs/frcnn

echo ""
echo "=== Training Complete ==="
echo "Model checkpoints saved to: tracking_baseline/train/runs/frcnn/"
echo ""
echo "To evaluate on test set, run:"
echo "  python eval_coco.py --checkpoint runs/frcnn/model_epoch6.pth --coco-json ../../data/coco_test.json --data-root ../../data"
