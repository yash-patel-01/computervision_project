from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import numpy as np
from .kalman import KalmanFilterCV
from ..utils.boxes import iou_matrix
from .association import cosine_distance, hungarian_assign, combine_cost

@dataclass
class Detection:
    bbox: List[float]  # xywh
    embedding: Optional[List[float]]
    class_id: int

class Track:
    def __init__(self, track_id: int, detection: Detection, kf: KalmanFilterCV, n_init: int, max_age: int):
        self.id = track_id
        self.kf = kf
        self.mean, self.cov = self.kf.initiate(self._measurement_from_bbox(detection.bbox))
        self.bbox = detection.bbox
        self.embedding = np.array(detection.embedding) if detection.embedding is not None else None
        self.class_id = detection.class_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0
        self.n_init = n_init
        self.max_age = max_age
        self.confirmed = False
    def _measurement_from_bbox(self, bbox):
        x, y, w, h = bbox
        cx = x + w / 2.0
        cy = y + h / 2.0
        a = w / h if h > 0 else 0.0
        return np.array([cx, cy, a, h], dtype=float)
    def predict(self):
        self.mean, self.cov = self.kf.predict(self.mean, self.cov)
        self.age += 1
        self.time_since_update += 1
    def update(self, detection: Detection):
        meas = self._measurement_from_bbox(detection.bbox)
        self.mean, self.cov = self.kf.update(self.mean, self.cov, meas)
        self.bbox = detection.bbox
        if detection.embedding is not None:
            emb = np.array(detection.embedding)
            if self.embedding is None:
                self.embedding = emb
            else:
                # Simple exponential moving average for embedding stability
                self.embedding = 0.9 * self.embedding + 0.1 * emb
        self.class_id = detection.class_id
        self.hits += 1
        self.time_since_update = 0
        if self.hits >= self.n_init:
            self.confirmed = True
    def is_deleted(self):
        return self.time_since_update > self.max_age

class Tracker:
    def __init__(self, cfg: Dict[str, Any]):
        self.tracks: List[Track] = []
        self.next_id = 1
        self.kf = KalmanFilterCV()
        self.appearance_weight = cfg.get('appearance_weight', 0.5)
        self.iou_weight = cfg.get('iou_weight', 0.5)
        self.max_cosine_dist = cfg.get('max_cosine_dist', 0.5)
        self.min_iou = cfg.get('min_iou', 0.3)
        self.max_age = cfg.get('max_age', 30)
        self.n_init = cfg.get('n_init', 3)
        self.embed_normalize = cfg.get('embed_normalize', True)
    def _prep_embeddings(self, detections: List[Detection]):
        embs = []
        for d in detections:
            if d.embedding is None:
                embs.append(None)
            else:
                arr = np.array(d.embedding, dtype=float)
                if self.embed_normalize:
                    norm = np.linalg.norm(arr) + 1e-8
                    arr = arr / norm
                embs.append(arr)
        return embs
    def predict(self):
        for t in self.tracks:
            t.predict()
    def update(self, detections: List[Detection]):
        # Separate confirmed and unconfirmed tracks for association (optional refinement later)
        if len(self.tracks) == 0:
            for d in detections:
                self._start_track(d)
            return
        track_bboxes = [t.bbox for t in self.tracks]
        det_bboxes = [d.bbox for d in detections]
        iou_cost = 1.0 - iou_matrix(det_bboxes, track_bboxes)  # lower is better
        # Appearance cost
        det_embs = self._prep_embeddings(detections)
        track_embs = [t.embedding for t in self.tracks]
        if all(e is None for e in det_embs) or all(e is None for e in track_embs):
            app_cost = None
        else:
            # Build embedding matrices where missing embeddings -> zeros (will be high distance)
            D_mat = []
            for e in det_embs:
                if e is None:
                    D_mat.append(np.zeros_like(track_embs[0]))
                else:
                    D_mat.append(e)
            T_mat = []
            for e in track_embs:
                if e is None:
                    T_mat.append(np.zeros_like(D_mat[0]))
                else:
                    T_mat.append(e)
            app_dist = cosine_distance(np.stack(D_mat), np.stack(T_mat))  # shape (Nd, Nt)
            app_cost = app_dist
        total_cost = combine_cost(iou_cost, app_cost, self.iou_weight, self.appearance_weight)
        assignment = hungarian_assign(total_cost)
        matched_det_idx = []
        unmatched_det_idx = []
        unmatched_track_idx = []
        for d_idx, t_idx in enumerate(assignment):
            if t_idx is None:
                unmatched_det_idx.append(d_idx)
            else:
                # gating
                if iou_cost[d_idx, t_idx] > (1.0 - self.min_iou):
                    unmatched_det_idx.append(d_idx)
                    unmatched_track_idx.append(t_idx)
                elif app_cost is not None and app_cost[d_idx, t_idx] > self.max_cosine_dist:
                    unmatched_det_idx.append(d_idx)
                    unmatched_track_idx.append(t_idx)
                else:
                    matched_det_idx.append((d_idx, t_idx))
        # Tracks not assigned by Hungarian
        assigned_track_indices = {t for _, t in matched_det_idx}
        for idx in range(len(self.tracks)):
            if idx not in assigned_track_indices and idx not in unmatched_track_idx:
                unmatched_track_idx.append(idx)
        # Update matched
        for d_i, t_i in matched_det_idx:
            self.tracks[t_i].update(detections[d_i])
        # Start new tracks for unmatched detections
        for d_i in unmatched_det_idx:
            self._start_track(detections[d_i])
        # Remove dead tracks
        self.tracks = [t for t in self.tracks if not t.is_deleted()]
    def _start_track(self, detection: Detection):
        tr = Track(self.next_id, detection, self.kf, self.n_init, self.max_age)
        self.tracks.append(tr)
        self.next_id += 1
    def active_tracks(self):
        return [t for t in self.tracks if t.confirmed]

# Helper to format output per frame
def format_tracks(tracks: List[Track]):
    out = []
    for t in tracks:
        x, y, w, h = t.bbox
        out.append({
            'track_id': t.id,
            'bbox': [x, y, w, h],
            'class_id': t.class_id,
            'age': t.age,
            'hits': t.hits
        })
    return out
