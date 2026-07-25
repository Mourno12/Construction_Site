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
# YOLO person detector. "yolo12n.pt" is auto-downloaded by ultralytics on
# first run if it isn't found locally. Swap in your own fine-tuned weights
# by pointing this at a custom .pt file.
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "yolo12n.pt")
YOLO_PERSON_CLASS_ID = 0          # COCO class id for "person"
YOLO_CONF_THRESHOLD = 0.45
YOLO_IOU_THRESHOLD = 0.45

# ByteTrack config shipped with ultralytics
TRACKER_CONFIG = "bytetrack.yaml"

# Fine-tuned ViT PPE classifier checkpoint directory (see train_vit.py).
# If this directory doesn't exist / doesn't load, the system automatically
# switches to the colour-heuristic classifier ("DEMO MODE").
VIT_CHECKPOINT_DIR = os.path.join(BASE_DIR, "models", "vit_ppe_classifier")
VIT_BASE_MODEL = "google/vit-base-patch16-224-in21k"
VIT_IMAGE_SIZE = 224

# Classification decision thresholds (sigmoid probability -> boolean)
HELMET_THRESHOLD = 0.5
VEST_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Heuristic (DEMO MODE) colour-based PPE fallback thresholds
# ---------------------------------------------------------------------------
HEURISTIC_HELMET_PIXEL_RATIO = 0.12   # fraction of head-crop pixels that must
                                       # match a "hard hat colour" to count as helmet
HEURISTIC_VEST_PIXEL_RATIO = 0.10

# ---------------------------------------------------------------------------
# Tracking (fallback IOU tracker, used when ultralytics ByteTrack unavailable)
# ---------------------------------------------------------------------------
TRACKER_MAX_MISSED_FRAMES = 30   # ~1s at 30fps before a track is dropped
TRACKER_MIN_IOU = 0.15           # lenient on purpose: the fallback HOG detector's boxes
                                   # jitter more between frames than YOLO's, so a strict
                                   # threshold here mints a fresh ID almost every frame

# ---------------------------------------------------------------------------
# Compliance rules
# ---------------------------------------------------------------------------
# A worker is "compliant" only if BOTH helmet and vest are detected.
REQUIRE_BOTH_FOR_COMPLIANCE = True

# A worker must be seen non-compliant for this many consecutive frames before
# an "alert" is emitted, to reduce flicker/false alarms.
ALERT_STREAK_FRAMES = 5

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
COLOR_TEXT_BG = (20, 20, 20)
COLOR_TEXT = (255, 255, 255)
