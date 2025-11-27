# Azure GPU Training Migration Guide

## Overview
This guide helps you move YOLO training from M1 Mac (19+ hours) to Azure GPU (~1-2 hours with T4/V100).

## What to Upload to Azure

### Required Files & Directories

```
├── data/
│   ├── yolo_dataset_full_abs/
│   │   ├── dataset.yaml
│   │   ├── images/
│   │   │   ├── train/  (symlinks - will need regeneration on Azure)
│   │   │   └── val/
│   │   └── labels/
│   │       ├── train/
│   │       └── val/
│   ├── coco_train.json
│   └── coco_test.json
├── tracking_baseline/
│   ├── src/
│   │   └── inference/
│   │       └── batch_ball_labeling.py
│   └── train/
│       └── runs/
│           └── yolo/
│               ├── train_yolo_ultra.py
│               ├── convert_coco_to_yolo.py
│               └── eval_coco_yolo.py
└── data/
    └── tracking-2023/  (original images for symlink regeneration)
```

### Upload Size Estimation
- **Images**: ~20-25 GB (train + val)
- **Labels**: ~50 MB
- **Scripts**: <1 MB
- **Total**: ~25 GB

## Step-by-Step Migration

### 1. Prepare Archive on Mac

```bash
cd /Users/yashpatel/Documents/Bocconi/Classes/Year\ 2/Semester\ 1/Deep\ Learning\ for\ Computer\ Vision/Project/Code

# Create archive (this will take 10-15 minutes)
tar -czf yolo_training_package.tar.gz \
  data/coco_train.json \
  data/coco_test.json \
  data/yolo_dataset_full_abs/dataset.yaml \
  data/yolo_dataset_full_abs/labels/ \
  data/tracking-2023/train/ \
  data/tracking-2023/test/ \
  tracking_baseline/train/runs/yolo/train_yolo_ultra.py \
  tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py \
  tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
  tracking_baseline/src/inference/batch_ball_labeling.py

# Check size
du -h yolo_training_package.tar.gz
```

### 2. Upload to Azure

**Option A: Azure Portal Upload**
1. Log into Azure Portal
2. Navigate to your VM or Storage Account
3. Use Azure Storage Explorer or portal upload
4. Upload `yolo_training_package.tar.gz`

**Option B: SCP (if SSH access configured)**
```bash
# Replace with your Azure VM details
scp yolo_training_package.tar.gz azureuser@<your-vm-ip>:~/
```

**Option C: Azure CLI**
```bash
az storage blob upload \
  --account-name <your-storage-account> \
  --container-name training-data \
  --name yolo_training_package.tar.gz \
  --file yolo_training_package.tar.gz
```

### 3. Setup on Azure VM

SSH into your Azure VM:
```bash
ssh azureuser@<your-vm-ip>
```

Extract and setup:
```bash
# Extract archive
tar -xzf yolo_training_package.tar.gz
cd <extracted-directory>

# Verify GPU availability
nvidia-smi

# Create conda environment (recommended)
conda create -n yolo_train python=3.10 -y
conda activate yolo_train

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ultralytics
pip install pycocotools
pip install Pillow
pip install tqdm

# Verify PyTorch sees GPU
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"
```

### 4. Generate the YOLO Dataset on Azure (Absolute Paths)

You MUST generate the YOLO-format dataset on Azure to ensure absolute paths match the VM's filesystem. Do NOT copy `yolo_dataset_full_abs` from Mac.

Preconditions:
- `data/tracking-2023/train` and `data/tracking-2023/test` exist on the Azure VM (from the uploaded archive).
- `data/coco_train.json` and `data/coco_test.json` exist (COCO annotations).

Generate Azure-local YOLO dataset:
```bash
# Create YOLO dataset with Azure-absolute paths
python tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py \
  --data-root data \
  --out-root data/yolo_dataset_azure \
  --workers 16 \
  --assume-width 1920 --assume-height 1080 \
  --progress-interval 1000

# Verify dataset.yaml exists and points to Azure paths
sed -n '1,20p' data/yolo_dataset_azure/dataset.yaml
ls -lh data/yolo_dataset_azure/images/train | head -20
ls -lh data/yolo_dataset_azure/labels/train | head -20
```

Notes:
- This step avoids broken symlinks and ensures Ultralytics can read images swiftly on Azure.
- If your raw images are in Azure Blob Storage, mount them or sync to `data/tracking-2023/*` first.

### 5. Launch Training (CUDA)

**Production Training Command:**
```bash
python tracking_baseline/train/runs/yolo/train_yolo_ultra.py \
  --dataset-root data/yolo_dataset_azure \
  --model yolov8n.pt \
  --epochs 50 \
  --batch 32 \
  --imgsz 832 \
  --name azure_ft_v1 \
  --lr0 0.005 \
  --patience 0 \
  --workers 8 \
  --mosaic 0.8 \
  --copy-paste 0.2 \
  --scale 0.7 \
  --translate 0.1 \
  --warmup-epochs 2 \
  --save-period 5 \
  --close-mosaic 10 \
  --cache ram \
  --device 0
```

**Quick Test (5 epochs, 2% data):**
```bash
python tracking_baseline/train/runs/yolo/train_yolo_ultra.py \
  --dataset-root data/yolo_dataset_azure \
  --model yolov8n.pt \
  --epochs 5 \
  --batch 32 \
  --imgsz 832 \
  --name azure_test \
  --lr0 0.005 \
  --workers 8 \
  --fraction 0.02 \
  --cache ram \
  --device 0
```

### 6. Monitor Training

**Real-time monitoring:**
```bash
# Watch progress
watch -n 5 "tail -20 tracking_baseline/train/runs/yolo/azure_ft_v1/results.csv"

# Check GPU usage
watch -n 2 nvidia-smi

# Monitor checkpoints
ls -lh tracking_baseline/train/runs/yolo/azure_ft_v1/weights/
```

**Estimated completion time:**
- T4 GPU: ~1.5-2 hours
- V100 GPU: ~1 hour
- A100 GPU: ~30-40 minutes

### 7. Evaluate Results

```bash
# Quick evaluation (100 images)
python tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
  --weights tracking_baseline/train/runs/yolo/azure_ft_v1/weights/best.pt \
  --data-root data \
  --limit 100

# Full evaluation
python tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
  --weights tracking_baseline/train/runs/yolo/azure_ft_v1/weights/best.pt \
  --data-root data
```

### 8. Download Results

```bash
# On Azure VM, create results package
cd tracking_baseline/train/runs/yolo
tar -czf azure_ft_v1_results.tar.gz azure_ft_v1/

# Download via SCP (from your Mac)
scp azureuser@<your-vm-ip>:~/tracking_baseline/train/runs/yolo/azure_ft_v1_results.tar.gz .
```

## Performance Comparison

| Platform | Batch | Device | Time/Epoch | Total (50 epochs) |
|----------|-------|--------|------------|-------------------|
| M1 Mac   | 16    | MPS    | ~23 min    | 19+ hours         |
| Azure T4 | 32    | CUDA   | ~2 min     | 1.5-2 hours       |
| Azure V100| 32   | CUDA   | ~1 min     | 1 hour            |
| Azure A100| 64   | CUDA   | ~30 sec    | 25-40 minutes     |

## Cost Estimation (Azure)

**Standard D4s v3 + T4 GPU:**
- Hourly cost: ~$1.50-2.00/hour
- Training time: 1.5-2 hours
- **Total cost: ~$3-4** for complete 50-epoch training

**NC6s v3 (V100):**
- Hourly cost: ~$3.00-3.50/hour
- Training time: ~1 hour
- **Total cost: ~$3-4**

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```bash
# Reduce batch size
--batch 16
# Or reduce image size
--imgsz 640
```

### Issue: Slow data loading
**Solution:**
```bash
# Increase workers
--workers 16
# Enable RAM caching
--cache ram
```

### Issue: symlinks broken after upload
**Solution:** Already handled - regenerate dataset with `convert_coco_to_yolo.py` on Azure.

### Issue: Training crashes overnight
**Solution:** Use `tmux` or `screen`:
```bash
# Start session
tmux new -s training

# Run training command
python tracking_baseline/train/runs/yolo/train_yolo_ultra.py ...

# Detach: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

## Alternative: Smaller Faster Experiment

If you want quick results to validate the approach first:

```bash
# 10 epochs, 10% of data (~10 minutes on T4)
python tracking_baseline/train/runs/yolo/train_yolo_ultra.py \
  --dataset-root data/yolo_dataset_azure \
  --model yolov8n.pt \
  --epochs 10 \
  --batch 32 \
  --imgsz 640 \
  --name azure_quick \
  --lr0 0.005 \
  --workers 8 \
  --fraction 0.1 \
  --cache ram \
  --device 0
```

This will give you a good indication if the training is working correctly before committing to the full 50-epoch run.

## Next Steps After Training

1. Download `best.pt` weights
2. Run full COCO evaluation
3. Compare to baseline (pretrained AP ~0.004)
4. Expected improvements: Player AP 0.3-0.5, Ball AP 0.1-0.2
5. Use weights for inference/tracking on test set

---

**Ready to migrate?** Start with Step 1 (creating the archive). Let me know if you need help with any specific step!
