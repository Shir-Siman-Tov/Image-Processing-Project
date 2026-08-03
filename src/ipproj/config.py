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
# FIGURES_ROOT is different: it always lives inside the repo itself and is
# NOT gitignored, because these PNGs are the README's visuals.

_REPO_ROOT = Path(__file__).resolve().parents[2]

try:
    import google.colab  # noqa: F401

    _IS_COLAB = True
except ImportError:
    _IS_COLAB = False

if _IS_COLAB:
    DATA_ROOT = Path("/content/drive/MyDrive/ipproj_data")
    CHECKPOINT_ROOT = Path("/content/drive/MyDrive/ipproj_checkpoints")
else:
    DATA_ROOT = _REPO_ROOT / "data"
    CHECKPOINT_ROOT = _REPO_ROOT / "checkpoints"

FIGURES_ROOT = _REPO_ROOT / "figures"  # tracked in git - README-ready PNGs, saved via viz.plotting.save_figure()

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
BILATERAL_FILTER_PARAMS = {"d": 9, "sigma_color": 75, "sigma_space": 75}
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
