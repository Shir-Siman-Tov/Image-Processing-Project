"""End-Point-Error for optical flow - the standard KITTI Flow 2015 evaluation metric."""
import numpy as np


def compute_epe(pred_flow: np.ndarray, gt_flow: np.ndarray, valid: np.ndarray) -> float:
    """pred_flow, gt_flow: (H, W, 2) arrays. valid: (H, W) bool mask of GT pixels to score."""
    error = np.linalg.norm(pred_flow - gt_flow, axis=-1)
    return float(error[valid].mean())
