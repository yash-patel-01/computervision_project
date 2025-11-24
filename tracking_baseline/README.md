# Football Player & Ball Tracking — Baseline

A tracking-by-detection baseline for football clips using Faster R-CNN detection and classical tracking algorithms.

Pipeline
- Detector (e.g., DETR) for players and ball
- Appearance embeddings per detection (use your provided vectors or learn a small re-ID head)
- Motion model: constant-velocity Kalman filter on bounding boxes
- Data association: blended cost (1 − IoU) + cosine distance; Hungarian matching (greedy fallback)
- Tracks update: matched → KF update; unmatched detections → new tracks; stale tracks → removed

What's included
- src/utils/boxes.py — IoU and box helpers
- src/tracking/kalman.py — lightweight Kalman filter (DeepSORT-style state: cx, cy, a, h + velocities)
- src/tracking/association.py — IoU/cosine distances and assignment (Hungarian or greedy)
- src/tracking/tracker.py — DeepSORT-like tracker that consumes boxes + embeddings
- src/inference/run_tracking.py — run tracker on precomputed detections to produce tracks
- configs/tracker.yaml — knobs for costs, thresholds, and lifecycle
- tests/test_iou.py — tiny sanity test

Inputs/Assumptions
- Detections per frame: bbox in xywh (top-left, width, height), class_id in {0: player, 1: ball}, embedding vector (list[float])
- If you don’t have embeddings yet, pass None and rely more on IoU costs

Quick start (after you have detections)
1) Put detections into a JSON file with this shape:
   {
     "frames": [
       {"frame_id": 0, "detections": [{"bbox": [x, y, w, h], "class_id": 0, "embedding": [..]}]},
       {"frame_id": 1, "detections": [...]},
       ...
     ]
   }
2) Run the tracker to produce tracks.json (IDs + boxes per frame).

Configuration
- configs/tracker.yaml controls:
  - appearance_weight, iou_weight
  - max_age, n_init, max_cosine_dist, min_iou
  - class handling (ball vs player)

Potential improvements
- Train detector at higher resolution to improve ball detection (AP_small)
- Add re-identification head with contrastive learning to reduce ID switches
- Experiment with temporal models (GRU) for more stable embeddings

Evaluation metrics
- Detection: mAP@[.5:.95], AP_small (important for ball detection)
- Tracking: IDF1, HOTA, MOTA, ID-switches

