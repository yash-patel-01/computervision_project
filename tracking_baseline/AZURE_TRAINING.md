# Azure VM Training Guide

## Prerequisites
- Azure VM with GPU (e.g., Standard_NC6s_v3 or similar)
- Ubuntu 20.04+ with CUDA drivers installed
- SSH access to the VM

## Setup Steps

### 1. Transfer Data to Azure VM

You'll need to upload your data folder to the VM. Options:

**Option A: Using SCP**
```bash
# From your local machine
scp -r "/Users/yashpatel/Documents/Bocconi/Classes/Year 2/Semester 1/Deep Learning for Computer Vision/Project/Code/tracking_baseline/data" azureuser@<VM-IP>:~/tracking_baseline/
```

**Option B: Azure Storage**
```bash
# Upload to Azure Blob Storage, then download from VM
az storage blob upload-batch -d mycontainer -s ./data
# On VM:
az storage blob download-batch -d ./data -s mycontainer
```

**Option C: Mount Azure File Share**
Mount an Azure Files share directly to the VM (recommended for large datasets).

### 2. Run Setup Script

```bash
ssh azureuser@<VM-IP>
cd tracking_baseline
chmod +x azure_setup.sh
./azure_setup.sh
```

### 3. Start Training

**Dry Run (validate pipeline):**
```bash
cd train
python3 train_frcnn.py \
  --limit 200 \
  --epochs 1 \
  --batch-size 4 \
  --log-interval 10 \
  --output-dir runs/frcnn_dryrun
```

**Full Training:**
```bash
python3 train_frcnn.py \
  --epochs 6 \
  --batch-size 8 \
  --lr 0.005 \
  --num-workers 4 \
  --output-dir runs/frcnn_full
```

### 4. Monitor Training

You can monitor training in real-time:
```bash
# View logs
tail -f runs/frcnn_full/train.log

# Check GPU usage
watch -n 1 nvidia-smi
```

### 5. Evaluate Model

After training completes:
```bash
python3 eval_coco.py \
  --checkpoint runs/frcnn_full/model_best.pth \
  --output runs/frcnn_full/eval_metrics.json
```

### 6. Download Results

```bash
# From local machine
scp azureuser@<VM-IP>:~/tracking_baseline/train/runs/frcnn_full/*.pth ./
scp azureuser@<VM-IP>:~/tracking_baseline/train/runs/frcnn_full/eval_metrics.json ./
```

## Expected Training Time

- Dry run (200 images, 1 epoch): ~5-10 minutes on GPU
- Full training (~6K images, 6 epochs): ~2-3 hours on single GPU (V100/T4)

## Troubleshooting

### Out of Memory
Reduce batch size:
```bash
python3 train_frcnn.py --batch-size 2 ...
```

### CUDA Not Available
Verify GPU drivers:
```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.is_available())"
```

### Data Path Issues
Ensure data structure matches:
```
tracking_baseline/
  data/
    coco_pseudo.json
    ball_tracks.json
    tracking-2023/
      train/
        SNMOT-XXX/
          img1/
      test/
        SNMOT-XXX/
          img1/
```

## Tips for Azure VMs

1. **Start/Stop VM**: Stop the VM when not training to save costs
2. **Use tmux**: Run training in tmux so it continues if SSH disconnects
   ```bash
   tmux new -s training
   # run training command
   # Detach: Ctrl+B then D
   # Reattach: tmux attach -t training
   ```
3. **Checkpoints**: Training saves checkpoints each epoch - you can resume with `--resume`
4. **Monitor costs**: Check Azure portal for VM usage costs

## Next Steps After Training

1. Evaluate model with `eval_coco.py` - focus on AP_small for ball detection
2. Use trained model for inference (generate detections JSON)
3. Run tracker with new detections
4. Compare tracking metrics vs pseudo-labeled baseline
