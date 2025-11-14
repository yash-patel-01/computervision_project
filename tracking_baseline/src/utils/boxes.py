import numpy as np


def xywh_to_x1y1x2y2(box):
    x, y, w, h = box
    return np.array([x, y, x + w, y + h], dtype=float)


def iou(box_a, box_b):
    """Compute IoU between two boxes in xywh format."""
    a = xywh_to_x1y1x2y2(box_a)
    b = xywh_to_x1y1x2y2(box_b)
    inter_x1 = max(a[0], b[0])
    inter_y1 = max(a[1], b[1])
    inter_x2 = min(a[2], b[2])
    inter_y2 = min(a[3], b[3])
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter_area
    if union <= 0.0:
        return 0.0
    return inter_area / union


def iou_matrix(dets, tracks):
    """Compute IoU matrix between detection boxes and track boxes (both xywh)."""
    if len(dets) == 0 or len(tracks) == 0:
        return np.zeros((len(dets), len(tracks)), dtype=float)
    M = np.zeros((len(dets), len(tracks)), dtype=float)
    for i, d in enumerate(dets):
        for j, t in enumerate(tracks):
            M[i, j] = iou(d, t)
    return M
