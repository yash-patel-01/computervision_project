from __future__ import annotations
import argparse
from pathlib import Path
import time
import datetime
import json

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from datasets.coco_ball import CocoBallDataset, collate_fn


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def build_model(num_classes: int, pretrained: bool = True):
    # Load a pre-trained model for classification and return
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=(
        torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.COCO_V1 if pretrained else None
    ))
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def default_transforms(img, target):
    # Convert PIL image to tensor; keep target unchanged
    return F.to_tensor(img), target


def train_one_epoch(model, optimizer, data_loader, device, epoch, log_interval=50):
    model.train()
    loss_sum = 0.0
    t0 = time.time()
    last_log_time = t0
    for i, (images, targets) in enumerate(data_loader):
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_value = float(losses.detach().cpu())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        loss_sum += loss_value
        # Log first few batches and then periodically
        if (i + 1) <= 10 or (i + 1) % log_interval == 0:
            avg = loss_sum / (i + 1)
            dt_since_log = time.time() - last_log_time
            batches_done = i + 1
            batches_total = len(data_loader)
            speed = dt_since_log / max((log_interval if batches_done > 10 else batches_done), 1)
            remaining_batches = batches_total - batches_done
            est_remaining = remaining_batches * speed
            eta = datetime.datetime.utcnow() + datetime.timedelta(seconds=est_remaining)
            print(
                f"Epoch {epoch} [{batches_done}/{batches_total}] loss={loss_value:.4f} avg={avg:.4f} "
                f"ETA ~{int(est_remaining//60)}m {int(est_remaining%60)}s (ETA UTC {eta.strftime('%H:%M:%S')})",
                flush=True
            )
            last_log_time = time.time()
    dt = time.time() - t0
    print(f"Epoch {epoch} done in {dt:.1f}s; avg loss {loss_sum/len(data_loader):.4f}", flush=True)


def save_checkpoint(model, optimizer, epoch, out_dir: Path, best: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / ("model_best.pth" if best else f"model_epoch{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, ckpt_path)
    return ckpt_path


def main():
    p = argparse.ArgumentParser(description="Train Faster R-CNN on pseudo-labeled COCO (player/ball)")
    cwd = Path(__file__).resolve().parent
    default_data_root = (cwd / ".." / ".." / ".." / ".." / "data").resolve()
    p.add_argument('--data-root', type=Path, default=default_data_root, help='Path to Project/Code/data')
    p.add_argument('--coco-json', type=Path, default=None, help='Path to coco_train.json; default under data/')
    p.add_argument('--epochs', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--momentum', type=float, default=0.9)
    p.add_argument('--weight-decay', type=float, default=0.0005)
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--limit', type=int, default=None, help='Optional: limit number of images for a quick dry-run')
    p.add_argument('--output-dir', type=Path, default=cwd.resolve())
    p.add_argument('--resume', type=Path, default=None)
    p.add_argument('--log-interval', type=int, default=50)
    p.add_argument('--dry-run', action='store_true', help='Load one batch and run a single forward/backward step, then exit')

    args = p.parse_args()
    device = get_device()
    print(f"Using device: {device}", flush=True)

    coco_json = args.coco_json or (args.data_root / 'coco_train.json')
    if not coco_json.is_file():
        raise FileNotFoundError(f"COCO annotations not found at {coco_json}")

    # Read categories to determine num_classes (include background)
    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    max_cat_id = max(cat['id'] for cat in coco_data['categories'])
    num_classes = max_cat_id + 1  # +1 for background as class 0
    print(f"Detected {len(coco_data['categories'])} categories; setting num_classes={num_classes}", flush=True)

    train_ds = CocoBallDataset(args.data_root, coco_json, transforms=default_transforms, limit=args.limit)
    print(f"Training on {len(train_ds)} images from train set", flush=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=(device.type != 'cpu')
    )

    model = build_model(num_classes=num_classes, pretrained=True)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=max(args.epochs // 3, 1), gamma=0.1)

    start_epoch = 1
    if args.resume and args.resume.is_file():
        ckpt = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt.get('epoch', 0) + 1
        print(f"Resumed from {args.resume}, epoch {start_epoch}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / 'config.json', 'w') as f:
        json.dump({
            'data_root': str(args.data_root),
            'coco_json': str(coco_json),
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'momentum': args.momentum,
            'weight_decay': args.weight_decay,
            'num_workers': args.num_workers,
            'limit': args.limit,
            'num_classes': num_classes
        }, f, indent=2)

    if args.dry_run:
        model.train()
        images, targets = next(iter(train_loader))
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(); losses.backward(); optimizer.step()
        print("Dry run complete: one batch forward/backward.")
        return

    for epoch in range(start_epoch, args.epochs + 1):
        train_one_epoch(
            model,
            optimizer,
            train_loader,
            device,
            epoch,
            log_interval=args.log_interval
        )
        lr_scheduler.step()
        ckpt_path = save_checkpoint(model, optimizer, epoch, args.output_dir, best=False)
        print(f"Saved checkpoint: {ckpt_path}", flush=True)

    print("Training complete.", flush=True)


if __name__ == '__main__':
    main()
