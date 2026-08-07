# Image-Processing-Project
Image Processing course final project: evaluating how classical computer-vision methods and a deep-learning model behave under image degradations (noise, blur, compression), and whether restoration or fine-tuning recovers the lost performance.

## Project Decisions

**Dataset: KITTI** — has native ground truth for 2D object detection, semantic segmentation, and optical flow.

**Tasks:**

| # | Task | Level | Method | GT source | Metric |
|---|------|-------|--------|-----------|--------|
| 1 | Feature/keypoint matching | low-level | ORB | none (self-referential correspondence) | match accuracy / repeatability |
| 2 | Optical flow | low-level | classical (Farneback) | KITTI Flow 2015 (native) | EPE / Fl-error |
| 3 | Object detection | high-level, DL | YOLOv8 (head fine-tuned on KITTI's 8 classes) | KITTI 2D Object (native) | mAP / IoU, per class |
| 4 | Semantic segmentation | high-level, DL | SegFormer (Cityscapes-pretrained) | KITTI Semantics, 200 imgs (native) | IoU |

**Distortions and their restoration counterparts:**

| Distortion | Restoration |
|---|---|
| Salt & Pepper noise | Median filter |
| Motion blur | Frequency-domain (Wiener) deconvolution |
| JPEG compression | Bilateral filtering + interpolation |

Full rationale (course-lecture grounding, the COCO→KITTI class-mismatch fix, SegFormer/Cityscapes class-ID alignment check) is documented in `CLAUDE.md`.

## Repository structure

```
src/ipproj/          reusable modules: config, datasets, tasks, distortions, restoration, metrics, viz
notebooks/            orchestration layer, run in order:
  01_dataset_visualization.ipynb
  02_clean_baseline.ipynb
  03_distortions.ipynb
  04_restoration.ipynb
  05_finetuning_distorted.ipynb
```

## Running on Colab

Each notebook's first cell mounts Google Drive and clones/installs this repo, so downloaded KITTI data and trained checkpoints persist across sessions. Run the notebooks in order — later notebooks depend on artifacts (the fine-tuned YOLO checkpoint, results CSVs) saved by earlier ones.

Each notebook's last cell also backs up `figures/`/`results/` to Drive and pushes them straight to GitHub (see "Figures" below) — that needs a GitHub personal access token (repo write scope), stored **once per Google account**:

1. Create a token at [github.com/settings/tokens](https://github.com/settings/tokens) (classic token, `repo` scope is enough).
2. In Colab, open the Secrets manager (key icon in the left sidebar), add a new secret named `GITHUB_TOKEN` with that token as the value, and enable "Notebook access".

Every notebook run then auto-commits/pushes any new figures/results straight to `main` — no manual download/`git add` step needed.

## Figures

Every plot in the notebooks is saved as a PNG under `figures/<notebook>/<name>.png` via `ipproj.viz.plotting.save_figure()`. Unlike `data/`/`checkpoints/` (gitignored, can be large), `figures/` is tracked in git. Each notebook's setup cell clones this repo fresh into a throwaway Colab filesystem, so a final cell (`ipproj.colab_sync`) copies `figures/` and `results/` into Drive as a backup and commits+pushes them back to GitHub before the runtime disconnects — see "Running on Colab" above for the one-time token setup. Embed a figure directly in this README with standard markdown, e.g.:

```markdown
![Baseline detection mAP per class](figures/02_clean_baseline/detection_map_per_class.png)
```

## Results

Per-notebook metric tables are saved as CSVs under `results/<notebook>/*.csv` via `ipproj.reporting.save_results_csv()` — tracked in git the same way `figures/` is, synced back from Colab by the same final-cell mechanism. Link/embed the relevant tables here as notebooks are run: baseline vs. distorted vs. restored vs. fine-tuned, per task and per class.
