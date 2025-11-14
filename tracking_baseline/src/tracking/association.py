import numpy as np
from typing import List, Optional

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False


def cosine_distance(A: np.ndarray, B: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Compute pairwise cosine distance between rows of A and B.
    Returns distances in [0, 2] where 0 means identical direction.
    """
    if A.size == 0 or B.size == 0:
        return np.zeros((A.shape[0], B.shape[0]), dtype=float)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + eps)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + eps)
    sim = A_norm @ B_norm.T
    # Convert similarity to distance
    return 1.0 - sim


def greedy_assign(cost: np.ndarray) -> List[Optional[int]]:
    """Greedy assignment: for each row, pick the lowest-cost unmatched column.
    Returns a list of assigned column indices (or None) per row.
    """
    n_rows, n_cols = cost.shape
    assigned_cols = set()
    assignment = [None] * n_rows
    for i in range(n_rows):
        j = int(np.argmin(cost[i])) if n_cols > 0 else None
        if j is not None and j not in assigned_cols:
            assignment[i] = j
            assigned_cols.add(j)
    return assignment


def hungarian_assign(cost: np.ndarray) -> List[Optional[int]]:
    if cost.size == 0:
        return []
    if SCIPY_AVAILABLE:
        r_idx, c_idx = linear_sum_assignment(cost)
        assignment = [None] * cost.shape[0]
        for r, c in zip(r_idx, c_idx):
            assignment[r] = int(c)
        return assignment
    else:
        return greedy_assign(cost)


def combine_cost(iou_cost: np.ndarray, app_cost: Optional[np.ndarray], w_iou: float, w_app: float) -> np.ndarray:
    if app_cost is None:
        return w_iou * iou_cost
    return w_iou * iou_cost + w_app * app_cost
