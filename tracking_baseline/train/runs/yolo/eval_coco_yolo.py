from __future__ import annotations
import argparse
from pathlib import Path
import json
from collections import defaultdict

from ultralytics import YOLO
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def run_eval(weights: Path, coco_json: Path, data_root: Path, imgsz: int = 640, limit: int = None) -> dict:
    model = YOLO(str(weights))
    coco = COCO(str(coco_json))
    cat_id_map = {cat['id']: idx for idx, cat in enumerate(coco.loadCats(coco.getCatIds()))}

    results = []
    img_ids = coco.getImgIds()
    if limit:
        img_ids = img_ids[:limit]
        print(f'Evaluating on {len(img_ids)} images (limited)')
    
    for idx, img_id in enumerate(img_ids):
        if (idx + 1) % 100 == 0:
            print(f'Processed {idx + 1}/{len(img_ids)} images...')
        img = coco.loadImgs([img_id])[0]
        file_path = data_root / img['file_name']
        preds = model.predict(source=str(file_path), imgsz=imgsz, verbose=False)
        if not preds:
            continue
        pred = preds[0]
        boxes = pred.boxes
        if boxes is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        for (x1, y1, x2, y2), s, c in zip(xyxy, conf, cls):
            x, y, w, h = float(x1), float(y1), float(x2 - x1), float(y2 - y1)
            results.append({
                'image_id': img_id,  # Keep as is (string or int)
                'category_id': int(c),
                'bbox': [x, y, w, h],
                'score': float(s)
            })

    if not results:
        print('Warning: No detections found')
        return {
            'AP': 0.0, 'AP50': 0.0, 'AP75': 0.0,
            'AP_small': 0.0, 'AP_medium': 0.0, 'AP_large': 0.0
        }
    
    # Ensure the coco dataset has required fields for loadRes
    if 'info' not in coco.dataset:
        coco.dataset['info'] = {}
    if 'licenses' not in coco.dataset:
        coco.dataset['licenses'] = []
    
    coco_dt = coco.loadRes(results)
    coco_eval = COCOeval(coco, coco_dt, iouType='bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    metrics = {
        'AP': float(coco_eval.stats[0]) if coco_eval.stats is not None else 0.0,
        'AP50': float(coco_eval.stats[1]) if coco_eval.stats is not None else 0.0,
        'AP75': float(coco_eval.stats[2]) if coco_eval.stats is not None else 0.0,
        'AP_small': float(coco_eval.stats[3]) if coco_eval.stats is not None else 0.0,
        'AP_medium': float(coco_eval.stats[4]) if coco_eval.stats is not None else 0.0,
        'AP_large': float(coco_eval.stats[5]) if coco_eval.stats is not None else 0.0,
    }
    return metrics


def main():
    p = argparse.ArgumentParser(description='Evaluate YOLOv8 weights on COCO-style JSON using COCOeval')
    cwd = Path(__file__).resolve().parent
    p.add_argument('--weights', type=str, default='yolov8n.pt', help='Path to YOLOv8 weights or model name (e.g., yolov8n.pt)')
    p.add_argument('--coco-test', type=Path, default=None, help='Path to coco_test.json')
    p.add_argument('--data-root', type=Path, default=(cwd.parent.parent.parent / "data").resolve())
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--limit', type=int, default=None, help='Limit number of test images for quick evaluation')
    p.add_argument('--out', type=Path, default=cwd / 'eval_metrics_yolo.json')
    args = p.parse_args()

    coco_test = args.coco_test or (args.data_root / 'coco_test.json')
    if not coco_test.is_file():
        raise FileNotFoundError(f"Missing test COCO JSON: {coco_test}")

    print(f'Evaluating {args.weights} on {coco_test}')
    metrics = run_eval(args.weights, coco_test, args.data_root, imgsz=args.imgsz, limit=args.limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nSaved metrics to {args.out}')
    print(f'AP: {metrics["AP"]:.3f}, AP50: {metrics["AP50"]:.3f}, AP_small: {metrics["AP_small"]:.3f}')


if __name__ == '__main__':
    main()
