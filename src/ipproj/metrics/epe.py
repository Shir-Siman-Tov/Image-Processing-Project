"""End-Point-Error and Fl-error for optical flow - the two standard KITTI Flow 2015 evaluation metrics."""
import numpy as np

from ipproj import config


def compute_epe(pred_flow: np.ndarray, gt_flow: np.ndarray, valid: np.ndarray) -> float:
    """pred_flow, gt_flow: (H, W, 2) arrays. valid: (H, W) bool mask of GT pixels to score."""
    if not valid.any():
        return 0.0
    error = np.linalg.norm(pred_flow - gt_flow, axis=-1)
    return float(error[valid].mean())


def compute_fl_error(pred_flow: np.ndarray, gt_flow: np.ndarray, valid: np.ndarray) -> float:
    """KITTI's Fl-all outlier rate: fraction of valid pixels whose EPE exceeds both
    config.FL_ERROR_PIXEL_THRESHOLD and config.FL_ERROR_RELATIVE_THRESHOLD of the GT flow magnitude.
    """
    if not valid.any():
        return 0.0
    error = np.linalg.norm(pred_flow - gt_flow, axis=-1)
    gt_magnitude = np.linalg.norm(gt_flow, axis=-1)
    outliers = (error > config.FL_ERROR_PIXEL_THRESHOLD) & (
        error > config.FL_ERROR_RELATIVE_THRESHOLD * gt_magnitude
    )
    return float(outliers[valid].mean())
