"""Shared plotting helpers - sample grids, GT overlays, before/after
comparisons, and performance curves/bars - reused across all 5 notebooks so
each one embeds consistent, README-ready figures rather than ad hoc plots.
"""
from pathlib import Path

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


def plot_metric_vs_intensity(intensity_values: list, metric_series: dict, xlabel: str, ylabel: str, title: str):
    """`metric_series`: {series_label: [metric value per intensity]}."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for label, values in metric_series.items():
        ax.plot(intensity_values, values, marker="o", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
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
