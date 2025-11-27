#!/bin/bash
# Quick archive creation script for Azure upload
# Run this from your project root directory

set -e  # Exit on error

echo "=== YOLO Training Azure Migration Package Creator ==="
echo ""

# Define base directory
BASE_DIR="/Users/yashpatel/Documents/Bocconi/Classes/Year 2/Semester 1/Deep Learning for Computer Vision/Project/Code"
cd "$BASE_DIR"

OUTPUT_FILE="yolo_training_package.tar.gz"

echo "Creating training package..."
echo "This will take 10-15 minutes due to ~25GB of image data"
echo ""

# Check if files exist before archiving
echo "Checking required files..."
required_files=(
    "data/coco_train.json"
    "data/coco_test.json"
    "tracking_baseline/train/runs/yolo/train_yolo_ultra.py"
    "tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py"
    "tracking_baseline/train/runs/yolo/eval_coco_yolo.py"
)

missing=0
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        missing=1
    else
        echo "✅ Found: $file"
    fi
done

if [ $missing -eq 1 ]; then
    echo ""
    echo "Error: Some required files are missing. Please check the paths."
    exit 1
fi

echo ""
echo "All required files found. Starting archive creation..."
echo ""

##
## Important: We DO NOT include the Mac's yolo_dataset_full_abs; it will be regenerated on Azure.
##

# Create the archive
tar -czf "$OUTPUT_FILE" \
    --exclude='*.cache' \
    --exclude='__pycache__' \
    --exclude='.DS_Store' \
    data/coco_train.json \
    data/coco_test.json \
    data/tracking-2023/train/ \
    data/tracking-2023/test/ \
    tracking_baseline/train/runs/yolo/train_yolo_ultra.py \
    tracking_baseline/train/runs/yolo/convert_coco_to_yolo.py \
    tracking_baseline/train/runs/yolo/eval_coco_yolo.py \
    tracking_baseline/src/inference/batch_ball_labeling.py \
    AZURE_MIGRATION_GUIDE.md \
    AZURE_QUICK_REFERENCE.md \
    azure_setup.sh

echo ""
echo "✅ Archive created successfully!"
echo ""
echo "Package details:"
ls -lh "$OUTPUT_FILE"
echo ""

# Calculate actual size
SIZE=$(du -h "$OUTPUT_FILE" | cut -f1)
echo "📦 Package size: $SIZE"
echo ""
echo "Next steps:"
echo "1. Upload $OUTPUT_FILE to your Azure VM or Storage Account"
echo "2. Follow instructions in AZURE_MIGRATION_GUIDE.md"
echo "3. Expected training time on Azure GPU: 1-2 hours (vs 19+ hours on M1)"
echo ""
