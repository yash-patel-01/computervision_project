from __future__ import annotations
import argparse
from pathlib import Path
import json

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.transforms import functional as F

from datasets.coco_ball import CocoBallDataset, collate_fn
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def default_transforms(img, target):
    return F.to_tensor(img), target


def build_model(num_classes: int, checkpoint: Path):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    ckpt = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(ckpt['model'])
    return model


def run_inference(model, data_loader, device):
    model.eval()
    results = []
    with torch.no_grad():
        for images, targets in data_loader:
            images = [img.to(device) for img in images]
            outputs = model(images)
            for target, output in zip(targets, outputs):
                image_id = int(target['image_id'])
                boxes = output['boxes'].cpu().numpy()
                scores = output['scores'].cpu().numpy()
                labels = output['labels'].cpu().numpy()
                for box, score, label in zip(boxes, scores, labels):
                    x1, y1, x2, y2 = box
                    w = x2 - x1
                    h = y2 - y1
                    results.append({
                        'image_id': image_id,
                        'category_id': int(label),
                        'bbox': [float(x1), float(y1), float(w), float(h)],
                        'score': float(score)
                    })
    return results


def evaluate(coco_gt_path: Path, detections: list[dict]):
    coco_gt = COCO(str(coco_gt_path))
    coco_dt = coco_gt.loadRes(detections)
    coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    # Extract AP_small (index 3 in stats) according to COCOeval docs
    metrics = {
        'AP': coco_eval.stats[0],
        'AP50': coco_eval.stats[1],
        'AP75': coco_eval.stats[2],
        'AP_small': coco_eval.stats[3],
        'AP_medium': coco_eval.stats[4],
        'AP_large': coco_eval.stats[5]
    }
    return metrics


def main():
    a = argparse.ArgumentParser(description='Evaluate Faster R-CNN checkpoint on test COCO')
    cwd = Path(__file__).resolve().parent
    default_data_root = (cwd / ".." / 'data').resolve()
    a.add_argument('--data-root', type=Path, default=default_data_root)
    a.add_argument('--coco-json', type=Path, default=None, help='Path to coco_test.json; default under data/')
    a.add_argument('--checkpoint', type=Path, required=True)
    a.add_argument('--batch-size', type=int, default=4)
    a.add_argument('--num-workers', type=int, default=4)
    a.add_argument('--limit', type=int, default=None)
    a.add_argument('--output', type=Path, default=(cwd / 'runs' / 'frcnn' / 'eval_metrics.json').resolve())

    args = a.parse_args()
    device = get_device()
    coco_json = args.coco_json or (args.data_root / 'coco_test.json')
    if not coco_json.is_file():
        raise FileNotFoundError(f"COCO annotations not found at {coco_json}")

    with open(coco_json, 'r') as f:
        coco_data = json.load(f)
    max_cat_id = max(cat['id'] for cat in coco_data['categories'])
    num_classes = max_cat_id + 1

    ds = CocoBallDataset(args.data_root, coco_json, transforms=default_transforms, limit=args.limit)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)

    model = build_model(num_classes=num_classes, checkpoint=args.checkpoint)
    model.to(device)

    detections = run_inference(model, loader, device)
    metrics = evaluate(coco_json, detections)
    print("Metrics:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics to {args.output}")


if __name__ == '__main__':
    main()
