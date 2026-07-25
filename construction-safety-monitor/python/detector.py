"""
detector.py
Person detector used as the first stage of the pipeline.

Primary path : YOLOv12n (Ultralytics) - attention-centric architecture,
               strong accuracy, class-filtered to "person". Note: per
               Ultralytics' own docs, YOLO12 trades some CPU throughput and
               training stability for accuracy vs YOLO11/YOLO26 (it leans on
               FlashAttention-style blocks) - great on a GPU-equipped demo
               box; if you're deploying to a CPU-only edge device, YOLO11n
               may run faster. Swap the weights in config.py at any time
               (e.g. YOLO_WEIGHTS="yolo11n.pt").
Fallback path: OpenCV's built-in HOG + Linear SVM person detector - runs
               anywhere OpenCV runs, no extra downloads, so the app still
               works if ultralytics/torch aren't installed yet.
"""

import sys
import numpy as np
import cv2

import config


def log(message, level="info"):
    import json
    print(json.dumps({"type": "log", "level": level, "message": message}), file=sys.stderr, flush=True)


class Detection:
    """A single detected person in one frame, pre-tracking."""

    __slots__ = ("bbox", "confidence")

    def __init__(self, bbox, confidence):
        self.bbox = bbox                # (x1, y1, x2, y2) ints
        self.confidence = confidence    # float 0-1


class YOLOPersonDetector:
    """Wraps an Ultralytics YOLO model, filtered to the 'person' class."""

    def __init__(self, weights=config.YOLO_WEIGHTS):
        from ultralytics import YOLO
        self.model = YOLO(weights)
        self.name = f"YOLOv12n ({weights})"

    def detect_and_track(self, frame):
        """
        Runs detection + ByteTrack tracking in a single call (Ultralytics
        handles association internally via `tracker=bytetrack.yaml`).
        Returns list of (track_id, Detection).
        """
        results = self.model.track(
            frame,
            persist=True,
            tracker=config.TRACKER_CONFIG,
            classes=[config.YOLO_PERSON_CLASS_ID],
            conf=config.YOLO_CONF_THRESHOLD,
            iou=config.YOLO_IOU_THRESHOLD,
            verbose=False,
        )
        out = []
        if not results:
            return out
        r = results[0]
        if r.boxes is None or len(r.boxes) == 0:
            return out
        ids = r.boxes.id
        for i, box in enumerate(r.boxes):
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            conf = float(box.conf[0]) if box.conf is not None else 0.0
            track_id = int(ids[i].item()) if ids is not None else -1
            out.append((track_id, Detection((x1, y1, x2, y2), conf)))
        return out


class HOGPersonDetector:
    """
    Fallback detector - no external weights, works out of the box.
    Does NOT do tracking itself; pair with tracker.SimpleIOUTracker.
    """

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.name = "OpenCV HOG (fallback - install ultralytics for YOLOv12n)"

    def detect(self, frame):
        boxes, weights = self.hog.detectMultiScale(
            frame, winStride=(8, 8), padding=(8, 8), scale=1.05
        )
        detections = []
        for (x, y, w, h), weight in zip(boxes, weights):
            conf = float(1 / (1 + np.exp(-weight)))  # squash HOG score to 0-1
            detections.append(Detection((int(x), int(y), int(x + w), int(y + h)), conf))
        return detections


def build_detector():
    """
    Attempts to load the real YOLOv12n detector. Falls back to HOG
    (no tracking built-in) if ultralytics/torch are unavailable or the
    weights fail to download/load.
    Returns: (detector, has_builtin_tracking: bool)
    """
    try:
        detector = YOLOPersonDetector()
        log(f"Loaded person detector: {detector.name}")
        return detector, True
    except Exception as e:
        log(f"Could not load YOLOv12n ({e}). Falling back to OpenCV HOG detector.", "warn")
        detector = HOGPersonDetector()
        log(f"Loaded person detector: {detector.name}", "warn")
        return detector, False
