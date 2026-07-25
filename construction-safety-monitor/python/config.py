"""
config.py
Central configuration for the Construction Site Safety Monitoring System.
Edit these values to point at your own trained weights, tune thresholds,
or change performance targets.
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Model weights
# ---------------------------------------------------------------------------
# YOLO detector. "yolo12n.pt" is auto-downloaded by ultralytics on first run
# if it isn't found locally. Swap in your own fine-tuned weights by pointing
# this at a custom .pt file.
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo12n.pt")
YOLO_PERSON_CLASS_ID = 0          # COCO class id for "person"
YOLO_CONF_THRESHOLD = 0.45
YOLO_IOU_THRESHOLD = 0.45

# Extra class id(s), beyond "person", to detect + track for the "object fall"
# feature (loose tools/materials/debris). Empty by default: the stock
# yolo12n.pt COCO weights have no generic "site object" class, so this only
# does anything once you fine-tune your own weights with such a class and
# either edit this list or set YOLO_OBJECT_CLASS_IDS="1,2" as an env var
# (comma-separated class ids). With this left empty, object-fall detection
# is simply inert - no object tracks ever exist to evaluate.
YOLO_OBJECT_CLASS_IDS = [
    int(c) for c in os.environ.get("YOLO_OBJECT_CLASS_IDS", "").split(",") if c.strip()
]

# ByteTrack config shipped with ultralytics
TRACKER_CONFIG = "bytetrack.yaml"

# ---------------------------------------------------------------------------
# PPE classifier labels (ViT + heuristic fallback both produce all of these)
# ---------------------------------------------------------------------------
# Order matters: it's the order of sigmoid outputs from the fine-tuned ViT
# head (see train_vit.py) and the column order expected in the training CSVs
# (see dataset.py). Add/remove labels here to retrain on a different set -
# nothing else in the pipeline hardcodes "5 PPE items + fallen".
PPE_LABELS = ["helmet", "vest", "gloves", "boots", "mask"]
ALL_TRAINABLE_LABELS = PPE_LABELS + ["fallen"]  # "fallen" is a per-crop worker-state label, trained the same way as PPE items

# Fine-tuned ViT PPE/state classifier checkpoint directory (see train_vit.py).
# If this directory doesn't exist / doesn't load, the system automatically
# switches to the colour + geometry heuristic classifier ("DEMO MODE").
VIT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "vit_ppe_classifier")
VIT_BASE_MODEL = "google/vit-base-patch16-224-in21k"
VIT_IMAGE_SIZE = 224

# Classification decision thresholds (sigmoid probability -> boolean), one
# per label in ALL_TRAINABLE_LABELS. 0.5 is a sane default for all of them;
# tune per-label here if one item runs hot/cold after training.
LABEL_THRESHOLDS = {label: 0.5 for label in ALL_TRAINABLE_LABELS}

# Which PPE items are actually mandatory for a worker to count as "compliant"
# (green/SAFE). Tracked-but-not-required items still show up in the worker
# log and per-item stats, they just don't flip someone to VIOLATION on their
# own. Defaults to all 5 - trim this list if e.g. gloves/mask aren't required
# on your site.
REQUIRED_PPE_ITEMS = list(PPE_LABELS)

# ---------------------------------------------------------------------------
# Heuristic (DEMO MODE) colour-based PPE fallback thresholds
# ---------------------------------------------------------------------------
# Fraction of the relevant crop region's pixels that must match the expected
# colour range to count as "present". These are deliberately approximate -
# see HeuristicPPEClassifier in ppe_classifier.py - real accuracy comes from
# training the ViT head (train_vit.py), not tuning these ratios forever.
HEURISTIC_HELMET_PIXEL_RATIO = 0.12   # head region
HEURISTIC_VEST_PIXEL_RATIO = 0.10     # torso region
HEURISTIC_GLOVES_PIXEL_RATIO = 0.08   # hand-side strips
HEURISTIC_BOOTS_PIXEL_RATIO = 0.10    # foot region
HEURISTIC_MASK_PIXEL_RATIO = 0.15     # lower-face region

# A standing person crop is taller than it is wide; once width/height climbs
# above this, the heuristic fallback treats them as fallen. This is a rough
# stand-in for the trained "fallen" label - it triggers on a person lying
# down, slumped sideways, etc. Real accuracy again comes from training the
# ViT head on labelled "fallen" crops.
FALL_ASPECT_RATIO_THRESHOLD = 1.3

# ---------------------------------------------------------------------------
# Tracking (fallback IOU tracker, used when ultralytics ByteTrack unavailable)
# ---------------------------------------------------------------------------
TRACKER_MAX_MISSED_FRAMES = 30   # ~1s at 30fps before a track is dropped
TRACKER_MIN_IOU = 0.15           # lenient on purpose: the fallback HOG detector's boxes
                                   # jitter more between frames than YOLO's, so a strict
                                   # threshold here mints a fresh ID almost every frame

# ---------------------------------------------------------------------------
# Compliance / alert rules
# ---------------------------------------------------------------------------
# A worker must be seen non-compliant (missing required PPE, or fallen) for
# this many consecutive frames before an alert is emitted, to reduce
# flicker/false alarms.
ALERT_STREAK_FRAMES = 5

# ---------------------------------------------------------------------------
# Object-fall detection (tracked objects only - see YOLO_OBJECT_CLASS_IDS)
# ---------------------------------------------------------------------------
# Downward speed (pixels/frame, on the resized MAX_FRAME_WIDTH frame) an
# object's tracked bbox centre must sustain to be considered "falling", held
# for this many consecutive frames before the alert fires. There's no static
# "does this image look like falling" model to train here - falling is a
# property of motion over time, not of a single frame - so this stays a
# physics-style rule on top of the (trainable) object detector/tracker.
OBJECT_FALL_MIN_DOWNWARD_PX_PER_FRAME = 12
OBJECT_FALL_STREAK_FRAMES = 4

# ---------------------------------------------------------------------------
# Performance
# ---------------------------------------------------------------------------
TARGET_FPS = 35
JPEG_QUALITY = 80          # quality for base64-encoded annotated frames sent to UI
MAX_FRAME_WIDTH = 960      # frames are downscaled to this width before inference

# ---------------------------------------------------------------------------
# Colours (BGR, for OpenCV drawing)
# ---------------------------------------------------------------------------
COLOR_SAFE = (94, 214, 61)      # green
COLOR_UNSAFE = (60, 60, 230)    # red
COLOR_WARN = (0, 184, 245)      # amber (single item missing)
COLOR_FALLEN = (200, 60, 220)   # magenta - worker-down, distinct from a plain PPE violation
COLOR_OBJECT = (255, 165, 0)    # orange - tracked non-person object
COLOR_OBJECT_FALL = (0, 60, 255)  # bright red-orange - active falling-object alert
COLOR_TEXT_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
