# Azure Training - Quick Reference Commands

## After uploading package to Azure VM

### 1. Extract Package
```bash
tar -xzf yolo_training_package.tar.gz
cd <extracted-directory>
```

### 2. Run Setup Script
```bash
bash azure_setup.sh
conda activate yolo_train  # if using conda
```

### 3. Generate Dataset on Azure (Required)
```bash
python tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py \
  --data-root data \
  --out-root data/yolo_dataset_azure \
  --workers 16 \
  --assume-width 1920 --assume-height 1080 \
  --progress-interval 1000
```
Expected time: 5-10 minutes on Azure GPU VM disks

Verify outputs:
```bash
sed -n '1,20p' data/yolo_dataset_azure/dataset.yaml
ls -lh data/yolo_dataset_azure/images/train | head -20
ls -lh data/yolo_dataset_azure/labels/train | head -20
```

### 4. Quick Sanity Test (Optional but Recommended)
```bash
# 5 epochs, 2% data, ~5 minutes
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

### 5. Full Production Training
```bash
# Using tmux to keep running if SSH disconnects
tmux new -s training

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

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t training
```

Expected time:
- T4 GPU: ~1.5-2 hours
- V100 GPU: ~1 hour
- A100 GPU: ~30-40 minutes

### 6. Monitor Progress

**Watch training metrics:**
```bash
watch -n 5 "tail -20 tracking_baseline/train/runs/yolo/azure_ft_v1/results.csv"
```

**Monitor GPU:**
```bash
watch -n 2 nvidia-smi
```

**Check saved checkpoints:**
```bash
ls -lh tracking_baseline/train/runs/yolo/azure_ft_v1/weights/
# Should see: epoch_5.pt, epoch_10.pt, ..., epoch_45.pt, best.pt, last.pt
```

### 7. Evaluate Results

**Quick evaluation (100 images):**
```bash
python tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
  --weights tracking_baseline/train/runs/yolo/azure_ft_v1/weights/best.pt \
  --data-root data \
  --limit 100
```

**Full evaluation:**
```bash
python tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
  --weights tracking_baseline/train/runs/yolo/azure_ft_v1/weights/best.pt \
  --data-root data
```

### 8. Download Results to Local Machine

**From Azure VM:**
```bash
cd tracking_baseline/train/runs/yolo
tar -czf azure_ft_v1_results.tar.gz azure_ft_v1/
```

**From your Mac:**
```bash
scp azureuser@<your-vm-ip>:~/path/to/azure_ft_v1_results.tar.gz .
```

Or just download the weights:
```bash
scp azureuser@<your-vm-ip>:~/path/to/azure_ft_v1/weights/best.pt ./yolo_azure_best.pt
```

---

## Troubleshooting

### Out of Memory Error
```bash
# Reduce batch size
--batch 16

# Or reduce image size
--imgsz 640
```

### Slow Training
```bash
# Make sure cache is enabled
--cache ram

# Increase workers
--workers 16
```

### Training Interrupted
```bash
# Resume from checkpoint
--resume
```

---

## Cost Optimization

**To minimize cost:**
1. Run quick test first (5 epochs, 2% data) - ~$0.10
2. If successful, run full training - ~$3-4
3. Shut down VM immediately after downloading results
4. Delete VM if you won't use it again soon

**Monitor costs:**
```bash
# Check how long training is taking
date  # Note start time
# After training completes:
date  # Note end time
# Calculate: (hours) × (VM hourly rate) = total cost
```

---

## Performance Expectations

**Baseline (pretrained yolov8n):**
- Overall AP: ~0.004
- Player AP: ~0.007
- Ball AP: 0.000

**Expected after fine-tuning:**
- Overall AP: 0.3-0.5
- Player AP: 0.4-0.6
- Ball AP: 0.1-0.2

**Massive improvement** especially for ball detection on small objects!
