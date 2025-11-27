from __future__ import annotations
import argparse
from pathlib import Path
import traceback
import json
import torch
from ultralytics import YOLO

def has_mps():
    return torch.backends.mps.is_available() and torch.backends.mps.is_built()

def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLOv8 on converted YOLO-format dataset (player, ball)")
    ap.add_argument('--dataset-root', type=Path, default=Path('data/yolo_dataset_full_abs'), help='Root containing dataset.yaml')
    ap.add_argument('--model', type=str, default='yolov8n.pt', help='Base model weights (.pt)')
    ap.add_argument('--epochs', type=int, default=2, help='Training epochs')
    ap.add_argument('--batch', type=int, default=16, help='Batch size')
    ap.add_argument('--imgsz', type=int, default=960, help='Image size for training/inference')
    ap.add_argument('--project', type=Path, default=Path('tracking_baseline/train/runs/yolo'), help='Ultralytics project dir')
    ap.add_argument('--name', type=str, default='ft_yolo', help='Run name')
    ap.add_argument('--lr0', type=float, default=0.001, help='Initial learning rate')
    ap.add_argument('--patience', type=int, default=0, help='Early stopping patience (0 disables)')
    ap.add_argument('--device', type=str, default=None, help='Override device (cpu, mps, 0, etc.)')
    ap.add_argument('--workers', type=int, default=4, help='Dataloader workers')
    ap.add_argument('--dry-run', action='store_true', help='Load dataset and single tiny epoch to validate config')
    # Augmentation / optimization flags
    ap.add_argument('--mosaic', type=float, default=1.0, help='Mosaic probability (0 disables)')
    ap.add_argument('--copy-paste', type=float, default=0.0, help='Copy-paste augmentation probability')
    ap.add_argument('--scale', type=float, default=0.5, help='Image scale gain range')
    ap.add_argument('--translate', type=float, default=0.1, help='Image translation fraction')
    ap.add_argument('--warmup-epochs', type=float, default=3.0, help='Number of warmup epochs')
    ap.add_argument('--save-period', type=int, default=0, help='Save checkpoint every N epochs (0 = only last/best)')
    ap.add_argument('--close-mosaic', type=int, default=10, help='Disable mosaic augmentation in last N epochs')
    ap.add_argument('--exist-ok', action='store_true', help='Allow existing run directory')
    ap.add_argument('--resume', action='store_true', help='Resume if checkpoint exists')
    ap.add_argument('--fraction', type=float, default=1.0, help='Use only a fraction of the dataset for rapid experiments (0<frac<=1).')
    ap.add_argument('--no-val', action='store_true', help='Disable validation during training (final evaluation run separately).')
    ap.add_argument('--cache', type=str, default='false', choices=['false','ram','disk'], help='Cache images to accelerate IO (ram may increase memory).')
    args = ap.parse_args()

    yaml_path = args.dataset_root / 'dataset.yaml'
    if not yaml_path.is_file():
        raise FileNotFoundError(f"dataset.yaml not found at {yaml_path}. Run convert_coco_to_yolo first.")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = 'cuda'
        elif has_mps():
            device = 'mps'
        else:
            device = 'cpu'
    print(f"Using device: {device}")

    model = YOLO(args.model)

    # Dry run check
    if args.dry_run:
        print('Dry run: initiating a single epoch preview (no saving).')
        model.train(
            data=str(yaml_path),
            epochs=1,
            batch=min(4, args.batch),
            imgsz=args.imgsz,
            device=device,
            project=str(args.project),
            name=f"{args.name}_dry",
            patience=0,
            save=False,
            verbose=True
        )
        print('Dry run complete.')
        return

    print('Starting fine-tuning...')
    out_dir = args.project / args.name
    out_dir.mkdir(parents=True, exist_ok=True)
    error_log = out_dir / 'error.log'

    try:
        cache_mode = False if args.cache == 'false' else args.cache
        results = model.train(
            data=str(yaml_path),
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=device,
            project=str(args.project),
            name=args.name,
            lr0=args.lr0,
            patience=args.patience,
            workers=args.workers,
            mosaic=args.mosaic,
            copy_paste=args.copy_paste,
            scale=args.scale,
            translate=args.translate,
            warmup_epochs=args.warmup_epochs,
            save_period=args.save_period if args.save_period > 0 else -1,
            close_mosaic=args.close_mosaic,
            exist_ok=args.exist_ok,
            resume=args.resume,
            fraction=args.fraction,
            val=not args.no_val,
            cache=cache_mode,
            verbose=True
        )
    except Exception as e:
        with open(error_log, 'w') as f:
            f.write('Training failed with exception:\n')
            f.write(str(e) + '\n')
            f.write('\nTraceback:\n')
            f.write(traceback.format_exc())
        print(f'Training failed. See {error_log}')
        return

    # Save training config
    config = {
        'model': args.model,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'lr0': args.lr0,
        'patience': args.patience,
        'device': device,
        'dataset_yaml': str(yaml_path)
    }
    with open(out_dir / 'train_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print(f'Training configuration saved to {out_dir / "train_config.json"}')
    print('Fine-tune complete.')

if __name__ == '__main__':
    main()
