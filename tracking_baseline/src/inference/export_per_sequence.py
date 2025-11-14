"""
Export per-sequence COCO and detections JSON using ball_tracks.json produced by batch_ball_labeling.py
- Input: data/ball_tracks.json (sequence -> list of ball track ids)
- For each sequence in train split, load gt/gt.txt to get tracks and write:
    data/per_seq/coco/coco_<SEQ>.json
    data/per_seq/dets/dets_<SEQ>.json
Note: test split typically lacks gt.txt, so this exporter focuses on train.
"""
from pathlib import Path
import json
from collections import defaultdict

DATA_ROOT = Path.cwd() / "data"
T23_ROOT = DATA_ROOT / "tracking-2023"
SPLITS = ["train", "test"]
OUT_COCO = DATA_ROOT / "per_seq" / "coco"
OUT_DETS = DATA_ROOT / "per_seq" / "dets"


def load_tracks(gt_path: Path):
    tracks = defaultdict(list)
    with open(gt_path) as f:
        for line in f:
            if not line.strip():
                continue
            fr, tid, x, y, w, h, conf, *_ = line.strip().split(',')
            fr_i = int(float(fr)); tid_i = int(float(tid))
            tracks[tid_i].append((fr_i, float(x), float(y), float(w), float(h)))
    return tracks


def export_coco_for_seq(seq_name: str, tracks: dict, ball_tids: list, out_path: Path):
    images = []
    annotations = []
    categories = [{"id":0,"name":"player"},{"id":1,"name":"ball"}]
    ann_id = 1
    frame_seen = set()
    for tid, boxes in tracks.items():
        for fr, x, y, w, h in boxes:
            if fr not in frame_seen:
                images.append({"id": fr, "file_name": f"{fr:06d}.jpg"})
                frame_seen.add(fr)
            cat_id = 1 if tid in ball_tids else 0
            annotations.append({
                "id": ann_id,
                "image_id": fr,
                "category_id": cat_id,
                "bbox": [x, y, w, h],
                "area": w * h,
                "iscrowd": 0,
                "track_id": tid,
            })
            ann_id += 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({"images": images, "annotations": annotations, "categories": categories}, f)


essential_dets_schema = {
    "frames": []
}

def export_dets_for_seq(seq_name: str, tracks: dict, ball_tids: list, out_path: Path):
    frame_map = defaultdict(list)
    for tid, boxes in tracks.items():
        for fr, x, y, w, h in boxes:
            cls_id = 1 if tid in ball_tids else 0
            frame_map[fr].append({"bbox": [x, y, w, h], "class_id": cls_id, "embedding": None})
    frames = [{"frame_id": fr, "detections": frame_map[fr]} for fr in sorted(frame_map.keys())]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump({"frames": frames}, f)


def main():
    ball_map_path = DATA_ROOT / "ball_tracks.json"
    if not ball_map_path.is_file():
        raise FileNotFoundError("data/ball_tracks.json not found. Run batch_ball_labeling.py first.")
    with open(ball_map_path, 'r') as f:
        ball_map = json.load(f)

    count = 0
    for split in SPLITS:
        split_root = T23_ROOT / split
        if not split_root.exists():
            continue
        for seq_dir in sorted(split_root.glob("SNMOT-*/")):
            seq_name = seq_dir.name
            if seq_name not in ball_map:
                # sequence may have been skipped during batch (e.g., overrides) or absent
                continue
            gt = seq_dir / "gt" / "gt.txt"
            if not gt.is_file():
                continue
            tracks = load_tracks(gt)
            ball_tids = ball_map.get(seq_name, [])
            # Write per-seq outputs
            export_coco_for_seq(seq_name, tracks, ball_tids, OUT_COCO / f"coco_{seq_name}.json")
            export_dets_for_seq(seq_name, tracks, ball_tids, OUT_DETS / f"dets_{seq_name}.json")
            count += 1
            print(f"[{split}] Exported per-seq files for {seq_name} (balls={len(ball_tids)})")
    print(f"Done. Wrote per-seq files for {count} sequences under {OUT_COCO.parent}")


if __name__ == "__main__":
    main()
