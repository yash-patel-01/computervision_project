from __future__ import annotations
import argparse
from pathlib import Path
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def load_coco(path: Path):
    with open(path, 'r') as f:
        return json.load(f)

def ensure_symlink(src: Path, dst: Path):
    """Create an absolute symlink. If an existing symlink points to a missing target, replace it."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    abs_src = src.resolve()
    if dst.is_symlink():
        try:
            current_target = dst.readlink()
            if not dst.exists() or not abs_src.is_file() or current_target != abs_src:
                dst.unlink(missing_ok=True)
        except OSError:
            dst.unlink(missing_ok=True)
    if not dst.exists():
        try:
            os.symlink(abs_src, dst)
        except FileExistsError:
            pass

def convert_split(
    coco_data: dict,
    split: str,
    out_root: Path,
    limit: int | None,
    data_root: Path,
    assume_size: tuple[int, int] | None,
    workers: int,
    progress_interval: int
):
    images = coco_data['images']
    anns = coco_data['annotations']
    cats = {c['id']: i for i, c in enumerate(sorted(coco_data['categories'], key=lambda x: x['id']))}

    if limit:
        images = images[:limit]
    img_id_set = {im['id'] for im in images}
    ann_by_img: dict = {}
    for a in anns:
        if a['image_id'] in img_id_set:
            ann_by_img.setdefault(a['image_id'], []).append(a)

    img_out_dir = out_root / 'images' / split
    lbl_out_dir = out_root / 'labels' / split
    img_out_dir.mkdir(parents=True, exist_ok=True)
    lbl_out_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    total = len(images)
    processed = 0

    def process_image(im):
        nonlocal processed
        img_id = im['id']
        file_rel = im['file_name']
        src_path = data_root / file_rel
        if not src_path.is_file():
            return False
        dst_path = img_out_dir / Path(file_rel).name
        ensure_symlink(src_path, dst_path)
        W = im.get('width')
        H = im.get('height')
        if (W is None or H is None):
            if assume_size is not None:
                W, H = assume_size
            else:
                from PIL import Image
                with Image.open(src_path) as pil_img:
                    W, H = pil_img.size
        yolo_lines = []
        for a in ann_by_img.get(img_id, []):
            x, y, w, h = a['bbox']
            cx = x + w / 2
            cy = y + h / 2
            cx_n = cx / W
            cy_n = cy / H
            w_n = w / W
            h_n = h / H
            cls = cats[a['category_id']]
            yolo_lines.append(f"{cls} {cx_n:.6f} {cy_n:.6f} {w_n:.6f} {h_n:.6f}")
        lbl_path = lbl_out_dir / (Path(file_rel).stem + '.txt')
        with open(lbl_path, 'w') as f:
            f.write('\n'.join(yolo_lines))
        processed += 1
        if progress_interval and processed % progress_interval == 0:
            elapsed = time.time() - start
            rate = processed / elapsed if elapsed > 0 else 0
            print(f"[{split}] {processed}/{total} images | {rate:.1f} img/s | ETA {(total-processed)/rate if rate>0 else 0:.1f}s")
        return True

    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(process_image, im) for im in images]
            for _ in as_completed(futures):
                pass
    else:
        for im in images:
            process_image(im)

    return processed, len(anns)

def write_dataset_yaml(out_root: Path, names: list[str]):
    yaml_path = out_root / 'dataset.yaml'
    content = [
        f"path: {out_root}",
        "train: images/train",
        "val: images/val",
        "names:",
    ]
    for i, n in enumerate(names):
        content.append(f"  {i}: {n}")
    yaml_path.write_text('\n'.join(content) + '\n')
    return yaml_path

def main():
    ap = argparse.ArgumentParser(description='Convert COCO (player/ball) to YOLO format with symlinked images.')
    ap.add_argument('--data-root', type=Path, default=Path('data'), help='Root containing images & coco JSONs')
    ap.add_argument('--coco-train', type=Path, default=None, help='Path to coco_train.json')
    ap.add_argument('--coco-val', type=Path, default=None, help='Path to coco_test.json (used as val)')
    ap.add_argument('--out-root', type=Path, default=Path('data/yolo_dataset_full_abs'), help='Output YOLO dataset root')
    ap.add_argument('--limit-train', type=int, default=None, help='Limit number of training images')
    ap.add_argument('--limit-val', type=int, default=None, help='Limit number of validation images')
    ap.add_argument('--assume-width', type=int, default=None, help='Assume fixed width if missing in JSON')
    ap.add_argument('--assume-height', type=int, default=None, help='Assume fixed height if missing in JSON')
    ap.add_argument('--workers', type=int, default=4, help='Thread workers for conversion')
    ap.add_argument('--progress-interval', type=int, default=100, help='Print progress every N images (0=disable)')
    args = ap.parse_args()

    coco_train = args.coco_train or (args.data_root / 'coco_train.json')
    coco_val = args.coco_val or (args.data_root / 'coco_test.json')
    if not coco_train.is_file() or not coco_val.is_file():
        raise FileNotFoundError('Missing coco_train.json or coco_test.json')

    train_data = load_coco(coco_train)
    val_data = load_coco(coco_val)
    names = [c['name'] for c in sorted(train_data['categories'], key=lambda x: x['id'])]

    args.out_root.mkdir(parents=True, exist_ok=True)
    yaml_path = write_dataset_yaml(args.out_root, names)

    assume_size = None
    if args.assume_width and args.assume_height:
        assume_size = (args.assume_width, args.assume_height)

    train_count, train_ann = convert_split(
        train_data, 'train', args.out_root, args.limit_train, args.data_root,
        assume_size, args.workers, args.progress_interval
    )
    val_count, val_ann = convert_split(
        val_data, 'val', args.out_root, args.limit_val, args.data_root,
        assume_size, args.workers, args.progress_interval
    )

    print(f"Converted train images: {train_count}, val images: {val_count}")
    print(f"Train annotations: {train_ann}, Val annotations: {val_ann}")
    print(f"Dataset YAML: {yaml_path}")
    print("Done.")

if __name__ == '__main__':
    main()
