"""Single source of truth for paths, seeds, and experiment parameters.

No distortion parameter, dataset URL, model version, or subset size should be
hardcoded anywhere outside this module (see CLAUDE.md's "no magic numbers" rule).
"""
from pathlib import Path

# --- Reproducibility -------------------------------------------------------

RANDOM_SEED = 42

# --- Storage roots -----------------------------------------------------------
# On Colab, mount Drive first (see each notebook's setup cell) and point
# DATA_ROOT/CHECKPOINT_ROOT there so large downloads/checkpoints survive
# session loss - these are gitignored, never meant to enter the repo.
# FIGURES_ROOT/REPORTS_ROOT are different: they always live inside the repo
# itself and are NOT gitignored, because these PNGs/CSVs are the README's
# visuals and result tables - see colab_sync.py for how they get from a
# throwaway Colab clone back onto GitHub.

REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import google.colab  # noqa: F401

    IS_COLAB = True
except ImportError:
    IS_COLAB = False

if IS_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/ipproj_data")
    CHECKPOINT_ROOT = Path("/content/drive/MyDrive/ipproj_checkpoints")
else:
    DATA_ROOT = REPO_ROOT / "data"
    CHECKPOINT_ROOT = REPO_ROOT / "checkpoints"

FIGURES_ROOT = REPO_ROOT / "figures"  # tracked in git - README-ready PNGs, saved via viz.plotting.save_figure()
REPORTS_ROOT = REPO_ROOT / "results"  # tracked in git - README-ready metric tables, saved via reporting.save_results_csv()

KITTI_ROOT = DATA_ROOT / "kitti"
DISTORTED_ROOT = DATA_ROOT / "distorted"
RESTORED_ROOT = DATA_ROOT / "restored"
RESULTS_ROOT = DATA_ROOT / "results"  # per-notebook metrics tables (CSV), persisted so later notebooks can load and compare

# --- KITTI download sources --------------------------------------------------
# Official KITTI benchmark archives (cvlibs.net / avg-kitti S3 mirror).
# Verify these are still current if a download fails - KITTI occasionally
# reorganizes its hosting.

KITTI_URLS = {
    "object_images": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip",
    "object_labels": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip",
    "semantics": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip",
    "scene_flow": "https://s3.eu-central-1.amazonaws.com/avg-kitti/data_scene_flow.zip",
}

# Small-scale subset sizes (per the course PDF's own "small scale, few
# examples" guidance) so notebooks stay Colab-friendly.
OBJECT_DETECTION_SUBSET_SIZE = 500
SEMANTIC_SEGMENTATION_SUBSET_SIZE = 200  # KITTI Semantics only has 200 labeled images total
OPTICAL_FLOW_SUBSET_SIZE = 200  # KITTI Flow 2015 only has 200 pairs total

TRAIN_VAL_TEST_SPLIT = (0.7, 0.15, 0.15)

# --- Task classes -------------------------------------------------------------
# KITTI's own 8 object classes (locked scope decision: fine-tune YOLO's head on
# these rather than trying to map COCO's 80 classes onto them). "DontCare" is a
# KITTI annotation convention marking ignore-regions, not a trainable class -
# excluded here, handled separately as an eval ignore-region.
KITTI_DETECTION_CLASSES = [
    "Car",
    "Van",
    "Truck",
    "Pedestrian",
    "Person_sitting",
    "Cyclist",
    "Tram",
    "Misc",
]
KITTI_IGNORE_LABEL = "DontCare"
# IoU overlap with a DontCare box above which a prediction is suppressed from
# scoring entirely (neither TP nor FP), matching KITTI's own eval protocol.
KITTI_IGNORE_IOU_THRESHOLD = 0.5

# --- Models -------------------------------------------------------------------
YOLO_BASE_WEIGHTS = "yolov8n.pt"  # COCO-pretrained backbone, head re-trained on KITTI_DETECTION_CLASSES
SEGFORMER_CHECKPOINT = "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"  # native class match with KITTI Semantics

# --- Distortions ----------------------------------------------------------
# Each distortion is swept across multiple intensity levels to satisfy the
# PDF's "measure performance per SNR, range of distortion intensities" requirement.

SALT_PEPPER_AMOUNTS = [0.01, 0.03, 0.05, 0.10, 0.20]  # fraction of pixels replaced
MOTION_BLUR_KERNEL_SIZES = [5, 9, 15, 21, 31]  # px, linear motion kernel length
JPEG_QUALITY_LEVELS = [80, 60, 40, 20, 10]  # lower = more compression artifacts

# --- Restoration parameters -----------------------------------------------

MEDIAN_FILTER_KERNEL_SIZE = 5
DECONVOLUTION_SNR_ESTIMATE = 0.01  # Wiener filter noise-to-signal assumption
BILATERAL_FILTER_PARAMS = {"d": 9, "sigmaColor": 75, "sigmaSpace": 75}  # cv2's Python binding keeps the C++ camelCase kwarg names
INTERPOLATION_SCALE = 2  # upsample factor before bilateral filtering, per the assignment's example

# --- Task hyperparameters ---------------------------------------------------

ORB_N_FEATURES = 1000
ORB_RATIO_TEST_THRESHOLD = 0.75  # Lowe's ratio test; deviates from the cited
# F_Features4ML lecture (which only describes brute-force greedy matching) -
# user-approved deviation, see feature_matching.py docstring.
FARNEBACK_PARAMS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}
# KITTI Flow 2015's own Fl-all criterion: a pixel is an outlier if its EPE
# exceeds both this pixel threshold and this fraction of the GT flow magnitude.
FL_ERROR_PIXEL_THRESHOLD = 3.0
FL_ERROR_RELATIVE_THRESHOLD = 0.05
YOLO_FINE_TUNE_EPOCHS = 50
YOLO_EVAL_BATCH_SIZE = 16  # Ultralytics inference batch size for evaluate()
YOLO_CONFIDENCE_THRESHOLD = 0.25  # score threshold for keeping a YOLO detection
SEGFORMER_FINE_TUNE_EPOCHS = 10
SEGFORMER_FINE_TUNE_LR = 5e-5

# Cityscapes' canonical 19 "trainId" classes, in order - KITTI Semantics reuses
# this exact taxonomy. Used by tasks/semantic_segmentation.py to verify the
# pretrained SegFormer checkpoint's logit order actually matches KITTI's GT
# mask IDs before trusting it (see the locked class-ID-alignment decision).
CITYSCAPES_TRAINID_LABELS = [
    "road", "sidewalk", "building", "wall", "fence", "pole", "traffic light",
    "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car",
    "truck", "bus", "train", "motorcycle", "bicycle",
]
# Populated only if tasks.semantic_segmentation.verify_class_alignment() finds
# the checkpoint's class order does NOT match CITYSCAPES_TRAINID_LABELS above -
# maps {checkpoint_class_id: kitti_trainid}. Leave None until that check runs.
SEGFORMER_CLASS_ID_REMAP = None

# Cityscapes' void/unlabeled trainId - also HF SegformerConfig's default
# semantic_loss_ignore_index, so eval metric and fine-tune loss share one value.
SEGFORMER_IGNORE_INDEX = 255

# KITTI Semantics' training/semantic/ masks store Cityscapes' raw 34-class
# label IDs (the cityscapesscripts labels.py "id" column) - NOT the 19-class
# "trainId" taxonomy CITYSCAPES_TRAINID_LABELS above uses. datasets.kitti's
# mask reader remaps id -> trainId (255 = void/ignore, matching
# SEGFORMER_IGNORE_INDEX) before any mask reaches the model/metric.
# Values cross-checked verbatim against {l.id: l.trainId for l in labels}
# from the official `cityscapesscripts` package (pip install cityscapesscripts;
# cityscapesscripts.helpers.labels) on 2026-08-09 - not hand-derived from memory.
# Not kept as a runtime dependency since this static table is all we need from it.
CITYSCAPES_ID_TO_TRAINID = {
    0: 255, 1: 255, 2: 255, 3: 255, 4: 255, 5: 255, 6: 255,
    7: 0, 8: 1, 9: 255, 10: 255,
    11: 2, 12: 3, 13: 4, 14: 255, 15: 255, 16: 255, 17: 5, 18: 255,
    19: 6, 20: 7,
    21: 8, 22: 9, 23: 10,
    24: 11, 25: 12, 26: 13, 27: 14, 28: 15, 29: 255, 30: 255,
    31: 16, 32: 17, 33: 18,
}
