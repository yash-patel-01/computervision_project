# Local Setup Instructions - Post Azure Migration

## Context
This project was developed on an Azure VM with GPU for YOLOv8 training. Training is complete and results have been pushed to GitHub. This document helps transition back to local development.

## What Was Done on Azure
1. **YOLOv8n Training**: 50 epochs on 42,000 images (10.15 hours)
   - Final metrics: mAP50=0.649, test AP=0.339
   - Best weights saved: `tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt` (6MB)
   - All results committed to GitHub repo `yash-patel-01/computervision_project`

2. **COCO to YOLO Conversion**: Generated 78,750 YOLO format labels
   - Location on Azure: `/home/Yash/Code/data/yolo_dataset_full_unique/`
   - Compressed archive available: `/tmp/yolo_dataset_full_unique.tar.gz` (30MB)
   - Contains: labels/, images/ (symlinks), dataset.yaml

3. **Files Pushed to GitHub**:
   - `tracking_baseline/train/runs/yolo/azure_full_v12/best.pt` - Trained weights (6MB)
   - `tracking_baseline/train/runs/yolo/azure_full_v12/TRAINING_SUMMARY.md` - Full documentation
   - `tracking_baseline/train/runs/yolo/azure_full_v12/results.csv` - All epoch metrics
   - `tracking_baseline/train/runs/yolo/azure_full_v12/eval_metrics_yolo_full_test.json` - Test results
   - Updated `.gitignore` and `convert_coco_to_yolo.py`

## Download Generated Data from Azure (Before VM Shutdown)

### Option 1: Download YOLO Dataset (30MB compressed)
```bash
# Download entire YOLO dataset folder
scp -P 5011 default@lab-563674bc-a073-4840-a789-8cf81e096250.westeurope.cloudapp.azure.com:/tmp/yolo_dataset_full_unique.tar.gz ./

# Extract locally
tar -xzf yolo_dataset_full_unique.tar.gz

# Move to project data directory
mv yolo_dataset_full_unique /path/to/your/local/Code/data/
```

### Option 2: Download Labels Only (28MB compressed)
```bash
# If you only need the YOLO labels (not symlinks)
scp -P 5011 default@lab-563674bc-a073-4840-a789-8cf81e096250.westeurope.cloudapp.azure.com:/tmp/yolo_labels.tar.gz ./

# Extract
tar -xzf yolo_labels.tar.gz
```

## Local Setup Steps

### 1. Pull Latest from GitHub
```bash
cd /path/to/your/local/Code
git pull origin main
```

This will give you:
- Trained YOLOv8 weights (`best.pt`)
- All training metrics and documentation
- Updated conversion scripts

### 2. Fix Image Symlinks (If Downloaded Dataset)
The image symlinks in `yolo_dataset_full_unique/images/` point to Azure paths and will be broken locally.

**Option A: Recreate symlinks to local images**
```bash
cd /path/to/your/local/Code/data/yolo_dataset_full_unique

# Remove broken symlinks
rm -rf images/

# Run conversion script again locally (it will recreate symlinks)
cd ../../tracking_baseline/train/runs/yolo
python convert_coco_to_yolo.py \
  --coco-train ../../../data/coco_train.json \
  --coco-test ../../../data/coco_test.json \
  --data-root ../../../data \
  --output ../../../data/yolo_dataset_full_unique_local
```

**Option B: Keep downloaded labels, create new dataset.yaml**
```bash
cd /path/to/your/local/Code/data

# Update dataset.yaml to point to local paths
cat > yolo_dataset_full_unique/dataset.yaml << EOF
path: /absolute/path/to/your/local/Code/data/yolo_dataset_full_unique
train: labels/train
val: labels/val
nc: 2
names: ['player', 'ball']
EOF
```

### 3. Local Environment Setup
```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
# or: venv\Scripts\activate  # On Windows

# Install dependencies
pip install torch torchvision ultralytics pycocotools pillow tqdm
```

### 4. Test Inference with Trained Model
```bash
cd /path/to/your/local/Code

# Run inference on a test image
python -c "
from ultralytics import YOLO
model = YOLO('tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt')
results = model.predict('data/tracking-2023/test/SNMOT-116/img1/000001.jpg', imgsz=832)
print('Inference successful!')
"
```

## Project Structure After Setup
```
Code/
├── data/
│   ├── tracking-2023/          # Original dataset (you have locally)
│   ├── jersey-2023/            # Original dataset (you have locally)
│   ├── coco_train.json         # COCO annotations (large)
│   ├── coco_test.json          # COCO annotations (large)
│   └── yolo_dataset_full_unique/  # Downloaded from Azure
│       ├── labels/             # YOLO format labels (THE IMPORTANT PART)
│       │   ├── train/          # 42,000 .txt files
│       │   └── val/            # 36,750 .txt files
│       ├── images/             # Symlinks (may need fixing)
│       └── dataset.yaml        # Config (may need path updates)
├── tracking_baseline/
│   └── train/
│       └── runs/
│           └── yolo/
│               ├── convert_coco_to_yolo.py
│               ├── train_yolo_ultra.py
│               ├── eval_coco_yolo.py
│               └── azure_full_v12/
│                   ├── weights/
│                   │   └── best.pt  # TRAINED MODEL (from GitHub)
│                   ├── TRAINING_SUMMARY.md
│                   ├── results.csv
│                   └── eval_metrics_yolo_full_test.json
└── .gitignore
```

## Common Issues & Solutions

### Issue 1: Broken Symlinks
**Problem**: `yolo_dataset_full_unique/images/` symlinks don't work locally
**Solution**: Either recreate symlinks with convert_coco_to_yolo.py (see above) OR just use the labels directly with your existing images in a new training setup

### Issue 2: Missing Dependencies
**Problem**: Import errors for torch, ultralytics, etc.
**Solution**: Install dependencies (see step 3 above)

### Issue 3: CUDA/GPU Not Available Locally
**Problem**: YOLOv8 training is slow on CPU
**Solution**: 
- Use the pre-trained weights from Azure for inference only
- For new training, consider using Google Colab or another cloud GPU
- Reduce batch size and image size for local CPU training (e.g., batch=8, imgsz=640)

### Issue 4: Can't Find Best Weights
**Problem**: `best.pt` not found after git pull
**Solution**: 
```bash
# Verify it's in the repo
ls -lh tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt

# If missing, check git
git status
git pull origin main
```

## What You Can Do Locally (Without GPU)

### ✅ Inference/Prediction (Fast on CPU)
```python
from ultralytics import YOLO
model = YOLO('tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt')

# Predict on single image
results = model.predict('path/to/image.jpg', imgsz=832)

# Predict on directory
results = model.predict('data/tracking-2023/test/SNMOT-116/img1/', imgsz=832)
```

### ✅ Evaluation
```bash
# Evaluate on test set (will take time on CPU)
python tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
  --weights tracking_baseline/train/runs/yolo/azure_full_v12/weights/best.pt \
  --coco-test data/coco_test.json \
  --data-root data \
  --imgsz 832 \
  --limit 100  # Limit to 100 images for faster testing
```

### ✅ Analysis & Visualization
- Review `TRAINING_SUMMARY.md` for complete training details
- Analyze `results.csv` for epoch-by-epoch metrics
- Use trained weights for inference and visualization

### ⚠️ Training (Slow on CPU)
- Not recommended locally without GPU
- Use cloud services (Colab, Azure, AWS) for new training runs

## Key Files to Keep Safe
1. **Trained weights**: `azure_full_v12/weights/best.pt` (6MB)
2. **YOLO labels**: `data/yolo_dataset_full_unique/labels/` (359MB uncompressed, 28MB compressed)
3. **Training results**: `azure_full_v12/results.csv` and `eval_metrics_yolo_full_test.json`
4. **Conversion script**: `convert_coco_to_yolo.py` (with unique filename fix)

## Next Steps
1. Pull from GitHub to get trained model
2. Download YOLO labels from Azure (before VM shutdown)
3. Fix symlinks if needed
4. Test inference locally
5. Continue development for tracking/analysis

## Azure VM Cleanup Checklist
Before shutting down Azure VM, ensure you have:
- [x] Pushed all code to GitHub
- [x] Downloaded `yolo_dataset_full_unique.tar.gz` (30MB)
- [ ] Optionally: Downloaded any other artifacts you need
- [ ] Verified `best.pt` is in your local repo after `git pull`

## Contact/Notes
- GitHub Repo: `yash-patel-01/computervision_project`
- Training completed: November 27, 2025
- Azure VM: lab-563674bc-a073-4840-a789-8cf81e096250.westeurope.cloudapp.azure.com (port 5011)
- Last commit: Updated .gitignore and added YOLOv8 training results
