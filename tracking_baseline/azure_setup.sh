#!/bin/bash
# Azure VM Setup Script for Faster R-CNN Training
# Run this after SSH'ing into your Azure VM with GPU

set -e

echo "=== Azure VM Training Setup ==="

# Update system packages
echo "Updating system packages..."
sudo apt-get update
sudo apt-get install -y python3-pip git

# Verify CUDA/GPU availability
echo "Checking GPU..."
nvidia-smi || echo "Warning: nvidia-smi not available. Ensure GPU drivers are installed."

# Clone or sync your repository (adjust repo URL as needed)
# git clone <your-repo-url> tracking_baseline
# cd tracking_baseline

# Install Python dependencies
echo "Installing Python packages..."
pip3 install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

# Verify PyTorch can see CUDA
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA device count: {torch.cuda.device_count()}')"

echo ""
echo "=== Setup Complete ==="
echo "To start training:"
echo "  cd train"
echo "  python3 train_frcnn.py --epochs 6 --batch-size 8 --output-dir runs/frcnn_gpu"
echo ""
echo "For dry run (200 images, 1 epoch):"
echo "  python3 train_frcnn.py --limit 200 --epochs 1 --batch-size 4 --output-dir runs/frcnn_dryrun"
