#!/bin/bash
# Azure VM Setup Script
# Run this script on your Azure VM after uploading and extracting the training package

set -e

echo "=== Azure VM YOLO Training Setup ==="
echo ""

# Check for GPU
echo "Checking GPU availability..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi
    echo ""
else
    echo "⚠️  Warning: nvidia-smi not found. Make sure you're on a GPU-enabled VM."
    echo ""
fi

# Check Python version
echo "Python version:"
python3 --version
echo ""

# Setup conda environment
echo "Setting up conda environment..."
if command -v conda &> /dev/null; then
    echo "Conda found. Creating environment..."
    conda create -n yolo_train python=3.10 -y
    
    # Activate instructions
    echo ""
    echo "✅ Conda environment created!"
    echo "To activate: conda activate yolo_train"
    echo ""
else
    echo "⚠️  Conda not found. Using system Python."
    echo "Consider installing Miniconda for better environment management."
    echo ""
fi

# Install PyTorch with CUDA
echo "Installing PyTorch with CUDA support..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other dependencies
echo "Installing additional dependencies..."
pip3 install ultralytics pycocotools Pillow tqdm

echo ""
echo "Verifying PyTorch CUDA setup..."
python3 -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

echo ""
echo "Verifying Ultralytics..."
python3 -c "import ultralytics; print(f'✅ Ultralytics version: {ultralytics.__version__}')"

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "1. Generate the YOLO dataset on Azure (absolute paths):"
echo "   python tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py \\\""
echo "     --data-root data \\\""
echo "     --out-root data/yolo_dataset_azure \\\""
echo "     --workers 16 \\\""
echo "     --assume-width 1920 --assume-height 1080 \\\""
echo "     --progress-interval 1000"
echo "   # Verify dataset:"
echo "   sed -n '1,20p' data/yolo_dataset_azure/dataset.yaml"
echo "   ls -lh data/yolo_dataset_azure/images/train | head -20"
echo "   ls -lh data/yolo_dataset_azure/labels/train | head -20"
echo ""
echo "2. Run quick test (2% data, 5 epochs, ~5 min):"
echo "   python tracking_baseline/train/runs/yolo/train_yolo_ultra.py \\"
echo "     --dataset-root data/yolo_dataset_azure \\"
echo "     --model yolov8n.pt --epochs 5 --batch 32 --imgsz 832 \\"
echo "     --name azure_test --fraction 0.02 --cache ram --device 0"
echo ""
echo "3. Run full training (50 epochs, ~1-2 hours):"
echo "   python tracking_baseline/train/runs/yolo/train_yolo_ultra.py \\"
echo "     --dataset-root data/yolo_dataset_azure \\"
echo "     --model yolov8n.pt --epochs 50 --batch 32 --imgsz 832 \\"
echo "     --name azure_ft_v1 --lr0 0.005 --patience 0 --workers 8 \\"
echo "     --mosaic 0.8 --copy-paste 0.2 --scale 0.7 --translate 0.1 \\"
echo "     --warmup-epochs 2 --save-period 5 --close-mosaic 10 \\"
echo "     --cache ram --device 0"
echo ""
