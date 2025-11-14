# Quick Start Guide

This guide will help you get started with the football tracking model in minutes.

## Installation

```bash
# Clone the repository
git clone https://github.com/yash-patel-01/computervision_project.git
cd computervision_project

# Install dependencies
pip install -r requirements.txt
```

## Prepare Your Data

1. **Organize your football clips into frames**
   ```bash
   mkdir -p data/train/images data/val/images
   # Extract frames from your videos and place them in these directories
   ```

2. **Create annotations in COCO format**
   - See `data/train/annotations.json` for an example
   - Use annotation tools like [CVAT](https://www.cvat.ai/) or [LabelImg](https://github.com/heartexlabs/labelImg)
   - Save as `data/train/annotations.json` and `data/val/annotations.json`

## Train the Model

```bash
# Basic training
python src/train.py \
    --train-data-dir data/train/images \
    --train-annotations data/train/annotations.json \
    --val-data-dir data/val/images \
    --val-annotations data/val/annotations.json \
    --batch-size 4 \
    --epochs 50 \
    --checkpoint-dir checkpoints

# Or use the config file
python src/train.py --config configs/default_config.yaml
```

Training will save checkpoints to the `checkpoints/` directory and the best model as `checkpoints/best_model.pth`.

## Run Inference

### On a Single Image
```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --image path/to/your/frame.jpg \
    --output result.jpg
```

### On a Video
```bash
python src/inference.py \
    --checkpoint checkpoints/best_model.pth \
    --video path/to/your/football_clip.mp4 \
    --output output_video.mp4
```

## Monitor Training

```bash
# In a separate terminal
tensorboard --logdir runs
# Open http://localhost:6006 in your browser
```

## Expected Results

The model will:
- Draw **green bounding boxes** around detected players
- Draw **red bounding boxes** around the detected ball
- Display confidence scores for each detection

## Tips for Best Results

1. **Data Quality**: Use high-resolution frames with clear visibility
2. **Annotation Accuracy**: Ensure bounding boxes are tight and accurate
3. **Training Duration**: More epochs generally lead to better results (50-100 epochs recommended)
4. **Confidence Threshold**: Adjust `--confidence-threshold` (0.3-0.7) based on your needs
5. **GPU Usage**: Use a CUDA-enabled GPU for faster training and inference

## Troubleshooting

| Problem | Solution |
|---------|----------|
| Out of memory | Reduce `--batch-size` to 2 or 1 |
| Slow training | Ensure you have a GPU available |
| Poor detections | Increase training epochs or add more training data |
| Missing dependencies | Run `pip install -r requirements.txt` |

## Next Steps

- Read the full [README.md](README.md) for detailed documentation
- Run `python example_usage.py` to see more examples
- Check the [configs/default_config.yaml](configs/default_config.yaml) for all available options

## Need Help?

- Check the example annotations in `data/train/annotations.json`
- Review the example usage with `python example_usage.py`
- Read the full documentation in README.md
