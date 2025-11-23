"""Batch pseudo-labeling of ball across tracking-2023 sequences.
Generates:
  - ball_tracks.json: {sequence_id: [ball_tid_1, ball_tid_2, ...] or []}
  - coco_train.json: COCO-style annotations for training sequences only
  - coco_test.json: COCO-style annotations for test sequences only
Heuristic: return ALL tracks that are small, square, and present (handles multiple balls per sequence).
Refine weights and threshold to our liking.
"""
from pathlib import Path
import json, statistics
from collections import defaultdict
import numpy as np

ROOT = Path.cwd() / "data" / "tracking-2023"
SPLITS = ["train", "test"]  # include both; 'challenge2023' intentionally skipped
OUT_DIR = Path.cwd() / "data"

AREA_PCT_THRESH = 30  # candidate must have median area below this percentile
ABS_AREA_MAX = 2000   # absolute area upper bound (px²) to exclude large players
MIN_ROUNDNESS = 0.7   # minimum roundness (aspect ratio near 1)
MAX_BALLS = 3         # max ball tracks per sequence


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

def stats_for_track(track_boxes):
    areas = [w*h for _,_,_,w,h in track_boxes]
    med_area = statistics.median(areas) if areas else 0.0
    aspect = [ (w/h) if h>0 else 0.0 for *_,w,h in track_boxes ]
    roundness = 1.0 - min(1.0, abs(statistics.median(aspect) - 1.0))
    # mean speed
    centers = sorted([(fr, x + w/2.0, y + h/2.0) for fr,x,y,w,h in track_boxes])
    dists = []
    for (_,x1,y1),(_,x2,y2) in zip(centers, centers[1:]):
        dists.append(((x2-x1)**2 + (y2-y1)**2)**0.5)
    mean_speed = float(np.mean(dists)) if dists else 0.0
    return med_area, roundness, mean_speed

def select_ball_tracks(tracks, min_obs=3):
    """Return list of up to MAX_BALLS track IDs that look like balls."""
    if not tracks:
        return [], {}
    stats = {tid: stats_for_track(boxes) for tid, boxes in tracks.items()}
    med_areas = np.array([s[0] for s in stats.values()])
    roundness_vals = np.array([s[1] for s in stats.values()])
    speed_vals = np.array([s[2] for s in stats.values()])
    # Extract presence_ratio from track stats
    presence_vals = []
    for tid_boxes in tracks.values():
        if tid_boxes:
            frames = [fr for fr,*_ in tid_boxes]
            presence = len(tid_boxes) / ((max(frames) - min(frames) + 1) if frames else 1)
            presence_vals.append(presence)
        else:
            presence_vals.append(0.0)
    presence_vals = np.array(presence_vals)
    
    # Normalize components
    def norm(v):
        v = np.array(v); return (v - v.min())/((v.max()-v.min()) or 1.0)
    med_norm = norm(med_areas)
    inv_area_score = 1.0 - med_norm  # prefer smaller
    speed_norm = norm(speed_vals)
    round_norm = norm(roundness_vals)
    presence_norm = norm(presence_vals)
    
    # Weighted combination
    combo = 0.50*inv_area_score + 0.30*presence_norm + 0.15*round_norm + 0.05*speed_norm
    
    area_thresh = np.percentile(med_areas, AREA_PCT_THRESH) if len(med_areas) else 0
    tids = list(stats.keys())
    candidates = []
    for idx, tid in enumerate(tids):
        med_a, roundness, _ = stats[tid]
        n_obs = len(tracks[tid])
        # Strict filters
        if med_a > min(area_thresh, ABS_AREA_MAX):
            continue  # too large (either relative or absolute)
        if n_obs < min_obs:
            continue  # too few observations
        if roundness < MIN_ROUNDNESS:
            continue  # not square enough
        candidates.append((tid, combo[idx]))
    # Sort by score descending, take top MAX_BALLS
    candidates.sort(key=lambda x: x[1], reverse=True)
    ball_tids = [tid for tid, _ in candidates[:MAX_BALLS]]
    return ball_tids, stats

def load_overrides():
        """Load optional overrides from tracking_baseline/data/overrides.json
        Structure per sequence name (e.g., "SNMOT-076"):
            {
                "SNMOT-076": {"skip": true},
                "SNMOT-061": {"ball_tids": [1,27]}
            }
        """
        try:
                base_dir = Path(__file__).resolve().parents[2]  # .../tracking_baseline
                overrides_path = base_dir / "data" / "overrides.json"
                if overrides_path.is_file():
                        with open(overrides_path, 'r') as f:
                                return json.load(f)
        except Exception:
                pass
        return {}

def build_coco(seqs_ball_map, seqs_tracks, seqs_split_map, target_split=None):
    """Build COCO dataset. If target_split is specified, only include sequences from that split."""
    images = []
    annotations = []
    categories = [{"id":0,"name":"player"},{"id":1,"name":"ball"}]
    ann_id = 1
    for seq_id, tracks in seqs_tracks.items():
        split = seqs_split_map.get(seq_id)
        if split is None:
            continue  # skip if not found
        if target_split is not None and split != target_split:
            continue  # skip if filtering by split
        
        ball_tids = seqs_ball_map.get(seq_id, [])
        # Collect all frames -> images
        frame_seen = set()
        for tid, boxes in tracks.items():
            for fr,x,y,w,h in boxes:
                if fr not in frame_seen:
                    images.append({"id": f"{seq_id}_{fr}", "file_name": f"tracking-2023/{split}/{seq_id}/img1/{fr:06d}.jpg"})
                    frame_seen.add(fr)
                cat_id = 1 if (tid in ball_tids) else 0
                annotations.append({
                    "id": ann_id,
                    "image_id": f"{seq_id}_{fr}",
                    "category_id": cat_id,
                    "bbox": [x,y,w,h],
                    "area": w*h,
                    "iscrowd": 0,
                    "track_id": tid
                })
                ann_id += 1
    return {"images": images, "annotations": annotations, "categories": categories}

def main():
    overrides = load_overrides()
    seqs_tracks = {}
    ball_map = {}
    seqs_split_map = {}  # Track which split each sequence belongs to
    
    for split in SPLITS:
        split_root = ROOT / split
        if not split_root.exists():
            continue
        for seq_dir in sorted(split_root.glob("SNMOT-*")):
            seq_name = seq_dir.name
            # Apply overrides: skip sequences or set manual ball_tids
            ov = overrides.get(seq_name, {}) if isinstance(overrides, dict) else {}
            if ov.get("skip", False):
                print(f"[{split}] {seq_name}: skipped by overrides")
                continue
            gt = seq_dir / "gt" / "gt.txt"
            if not gt.is_file():
                # Some splits (e.g., challenge2023) may lack GT; skip silently
                continue
            tracks = load_tracks(gt)
            if "ball_tids" in ov:
                ball_tids = list(ov["ball_tids"]) or []
                print(f"[{split}] {seq_name}: ball_tids={ball_tids} (from overrides)")
            else:
                ball_tids, stats = select_ball_tracks(tracks)
                print(f"[{split}] {seq_name}: ball_tids={ball_tids}")
            seqs_tracks[seq_name] = tracks
            ball_map[seq_name] = ball_tids
            seqs_split_map[seq_name] = split
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / "ball_tracks.json", 'w') as f:
        json.dump(ball_map, f)
    
    # Generate separate COCO files for train and test
    coco_train = build_coco(ball_map, seqs_tracks, seqs_split_map, target_split="train")
    with open(OUT_DIR / "coco_train.json", 'w') as f:
        json.dump(coco_train, f)
    
    coco_test = build_coco(ball_map, seqs_tracks, seqs_split_map, target_split="test")
    with open(OUT_DIR / "coco_test.json", 'w') as f:
        json.dump(coco_test, f)
    
    print(f"Saved ball_tracks.json, coco_train.json ({len(coco_train['images'])} images), and coco_test.json ({len(coco_test['images'])} images)")

if __name__ == "__main__":
    main()
