"""Shared plotting helpers - sample grids, GT overlays, before/after
comparisons, and performance curves/bars - reused across all 5 notebooks so
each one embeds consistent, README-ready figures rather than ad hoc plots.
"""
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from ipproj import config


def save_figure(fig, relative_path: str) -> Path:
    """Saves `fig` as PNG under config.FIGURES_ROOT/relative_path, creating
    parent directories as needed. `relative_path` should include the .png
    extension, e.g. "01_dataset_visualization/detection_gt_grid.png".

    Unlike config.DATA_ROOT/CHECKPOINT_ROOT, FIGURES_ROOT lives inside the
    repo and is NOT gitignored - these are meant to be committed and embedded
    directly in the README (`![caption](figures/...png)`).
    """
    out_path = config.FIGURES_ROOT / relative_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path


def plot_image_grid(images: list, titles: list = None, ncols: int = 4, figsize: tuple = None):
    nrows = (len(images) + ncols - 1) // ncols
    figsize = figsize or (4 * ncols, 4 * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            if titles:
                ax.set_title(titles[i], fontsize=10)
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_detection_boxes(image: np.ndarray, boxes: np.ndarray, classes: list, ax=None, color: str = "lime"):
    ax = ax or plt.gca()
    ax.imshow(image)
    for (x1, y1, x2, y2), cls in zip(boxes, classes):
        ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=color, linewidth=1.5))
        ax.text(x1, y1 - 2, cls, color=color, fontsize=8, backgroundcolor="black")
    ax.axis("off")
    return ax


def plot_segmentation_mask(image: np.ndarray, mask: np.ndarray, ax=None, alpha: float = 0.5):
    ax = ax or plt.gca()
    ax.imshow(image)
    ax.imshow(mask, alpha=alpha, cmap="tab20")
    ax.axis("off")
    return ax


def plot_orb_keypoints(image: np.ndarray, keypoints, ax=None, color: tuple = (0, 255, 0)):
    ax = ax or plt.gca()
    rendered = cv2.drawKeypoints(image, keypoints, None, color=color, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    ax.imshow(rendered)
    ax.axis("off")
    return ax


def flow_to_color(flow: np.ndarray) -> np.ndarray:
    """HSV-encodes a dense flow field for display: hue = direction, value =
    magnitude (normalized to the frame's own range). Standard OpenCV
    visualization convention - purely a rendering choice, not part of the
    EPE/Fl-error metric computation."""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((*flow.shape[:2], 3), dtype=np.uint8)
    hsv[..., 0] = angle * 180 / np.pi / 2
    hsv[..., 1] = 255
    hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def plot_optical_flow(flow: np.ndarray, ax=None):
    ax = ax or plt.gca()
    ax.imshow(flow_to_color(flow))
    ax.axis("off")
    return ax


def plot_before_after(before: np.ndarray, after: np.ndarray, title_before: str = "Before", title_after: str = "After"):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    axes[0].imshow(before)
    axes[0].set_title(title_before)
    axes[0].axis("off")
    axes[1].imshow(after)
    axes[1].set_title(title_after)
    axes[1].axis("off")
    fig.tight_layout()
    return fig


def plot_metric_vs_intensity(
    intensity_values: list, metric_series: dict, xlabel: str, ylabel: str, title: str, invert_xaxis: bool = False,
):
    """`metric_series`: {series_label: [metric value per intensity]}.

    `invert_xaxis`: flip the x-axis direction, e.g. for an SNR (dB) x-axis
    where higher = less distortion - inverting makes distortion severity
    increase left-to-right, matching how the other curves in this module
    read. Off by default since most callers use a naturally-ascending x-axis
    (epoch, distortion intensity level)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in metric_series.items():
        ax.plot(intensity_values, values, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if invert_xaxis:
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def plot_metric_vs_snr_by_distortion(
    snr_by_distortion: dict, metric_by_distortion: dict, clean_reference: float, xlabel: str, ylabel: str, title: str,
    invert_xaxis: bool = False,
):
    """One curve per distortion, each with its own SNR x-values (distortions
    degrade SNR at different rates, so a shared x-axis like
    `plot_metric_vs_intensity` doesn't fit). `clean_reference`: optional
    float, drawn as a horizontal dashed line; pass None to omit it (e.g.
    metrics with no meaningful clean-only baseline). `invert_xaxis`: see
    `plot_metric_vs_intensity`."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, snr_values in snr_by_distortion.items():
        metric_values = metric_by_distortion[name]
        order = np.argsort(snr_values)
        ax.plot(np.array(snr_values)[order], np.array(metric_values)[order], marker="o", label=name)
    if clean_reference is not None:
        ax.axhline(clean_reference, linestyle="--", color="gray", label="clean baseline")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    if invert_xaxis:
        ax.invert_xaxis()
    fig.tight_layout()
    return fig


def plot_bar_per_class(class_names: list, values: list, ylabel: str, title: str):
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 0.8), 4))
    ax.bar(class_names, values)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def plot_grouped_bar_per_class(
    class_names: list, series: dict, ylabel: str, title: str, show_values: bool = False,
    reference_value: float = None, reference_label: str = "clean baseline",
):
    """`series`: {series_label: [value per class, aligned to class_names]}. One
    cluster of bars per class, one bar per series within each cluster - e.g.
    clean vs. a specific distortion severity, side by side per class.

    `show_values`: annotate each bar with its numeric value. Off by default -
    it clutters charts with many classes (e.g. 19 Cityscapes labels), but is
    useful for small category counts like a per-task clean/distorted/restored
    summary.

    `reference_value`: optional float, drawn as a horizontal dashed line
    (same convention as plot_metric_bars_by_condition) - for a value that's
    constant across all classes/conditions (e.g. a clean baseline), a line is
    clearer than a repeated identical bar in every cluster. Pass None to omit
    it (default), which also keeps every existing call site unchanged."""
    fig, ax = plt.subplots(figsize=(max(6, len(class_names) * 1.2), 4))
    n_series = len(series)
    bar_width = 0.8 / n_series
    x = np.arange(len(class_names))
    for i, (label, values) in enumerate(series.items()):
        offset = (i - (n_series - 1) / 2) * bar_width
        bars = ax.bar(x + offset, values, bar_width, label=label)
        if show_values:
            ax.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
    if reference_value is not None:
        ax.axhline(
            reference_value, linestyle="--", color="#d03b3b",
            label=f"{reference_label} = {reference_value:.3f}",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    return fig


def plot_metric_bars_by_condition(
    condition_labels: list, values: list, bar_colors: list, ylabel: str, title: str,
    reference_value: float = None, reference_label: str = "clean baseline",
):
    """Bar per (clean + distortion x severity level) condition, with an
    optional horizontal dashed line for the clean baseline - the
    severity-axis counterpart to plot_metric_vs_snr_by_distortion's SNR-axis
    reference line. `bar_colors`: one color per condition, e.g. gray for
    "clean" and a fixed color per distortion family so groups read at a
    glance."""
    fig, ax = plt.subplots(figsize=(max(8, len(condition_labels) * 0.55), 4.5))
    ax.bar(condition_labels, values, color=bar_colors)
    if reference_value is not None:
        ax.axhline(
            reference_value, linestyle="--", color="#d03b3b",
            label=f"{reference_label} = {reference_value:.3f}",
        )
        ax.legend()
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig
