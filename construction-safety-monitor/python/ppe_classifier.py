"""
ppe_classifier.py
Classifies a cropped worker image for PPE compliance (helmet, vest, gloves,
boots, mask) and worker state (fallen), one boolean + confidence per label
in config.ALL_TRAINABLE_LABELS.

Two independent fine-tuned ViT checkpoints back this, not one - see the
comment on VIT_PPE_CHECKPOINT_DIR/VIT_FALL_CHECKPOINT_DIR in config.py for
why. Either one missing/untrained falls back to the colour+geometry
heuristic for just that piece, independently:
  - PPE items (helmet/vest/gloves/boots/mask): HSV colour masks over the
    crop region where that item would appear on a standing worker.
  - fallen: the crop's aspect ratio (a lying-down/slumped worker's bounding
    box is wider than it is tall, unlike a standing one).
No training/weights required for the heuristic path, so the whole pipeline
is runnable immediately after `git clone`.
"""

import os
import sys
import json
import numpy as np
import cv2

import config


def log(message, level="info"):
    print(json.dumps({"type": "log", "level": level, "message": message}), file=sys.stderr, flush=True)


class PPEResult:
    """
    Holds one boolean + confidence per label in config.ALL_TRAINABLE_LABELS
    (by default: helmet, vest, gloves, boots, mask, fallen). Both classifier
    implementations below produce the same shape, so nothing downstream
    needs to know which one is active.
    """

    __slots__ = ("values", "confidences")

    def __init__(self, values, confidences):
        self.values = values            # {"helmet": True, ...}
        self.confidences = confidences  # {"helmet": 0.92, ...}

    def __getattr__(self, name):
        # Lets callers use ppe.helmet / ppe.fallen / etc. directly, without
        # this class needing an explicit property per label.
        try:
            return self.values[name]
        except KeyError:
            raise AttributeError(name)

    @property
    def compliant(self):
        # Fallen overrides everything else - a worker who's down is not
        # "safe" no matter what PPE they have on.
        if self.values.get("fallen", False):
            return False
        return all(self.values.get(item, False) for item in config.REQUIRED_PPE_ITEMS)

    def to_dict(self):
        d = dict(self.values)
        d.update({f"{label}Confidence": round(conf, 3) for label, conf in self.confidences.items()})
        d["compliant"] = self.compliant
        return d


class ViTLabelClassifier:
    """
    Fine-tuned Vision Transformer classifier for one specific label set
    (either config.PPE_LABELS or config.FALL_LABELS) - a thin, reusable
    wrapper so the PPE model and the fall model load/run identically
    without duplicating this logic twice.
    """

    def __init__(self, checkpoint_dir, labels):
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor
        from device_utils import select_device

        self.torch = torch
        self.labels = labels
        self.device = select_device()
        self.processor = ViTImageProcessor.from_pretrained(checkpoint_dir)
        self.model = ViTForImageClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()
        self.name = f"Fine-tuned ViT ({checkpoint_dir}) on {self.device}"

    def classify_raw(self, crop_bgr):
        """Returns {label: (value, confidence)} for just this model's labels."""
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits  # shape (1, len(self.labels))
            probs = self.torch.sigmoid(logits)[0].cpu().numpy()

        out = {}
        for label, prob in zip(self.labels, probs):
            prob = float(prob)
            out[label] = (prob >= config.LABEL_THRESHOLDS[label], prob)
        return out


class HeuristicPPEClassifier:
    """
    DEMO MODE fallback - no trained weights required.
    Approximates each PPE label using HSV colour masks over the crop region
    where that item would appear on a standing worker, and flags "fallen"
    from the crop's aspect ratio (a lying-down/slumped worker's bounding box
    is wider than it is tall, unlike a standing one). This is intentionally
    simple so it runs everywhere; swap in the real ViT model (train_vit.py)
    for production-grade accuracy.
    """

    def __init__(self):
        self.name = "Colour+geometry heuristic classifier (DEMO MODE - train/plug in ViT for real accuracy)"
        # HSV ranges (OpenCV: H 0-179, S 0-255, V 0-255)
        self._helmet_ranges = [
            ((18, 80, 120), (35, 255, 255)),   # yellow/orange hard hat
            ((0, 0, 200), (179, 40, 255)),      # white hard hat
            ((0, 100, 100), (10, 255, 255)),    # red hard hat
        ]
        self._vest_ranges = [
            ((20, 100, 120), (35, 255, 255)),   # hi-vis yellow/lime
            ((5, 120, 120), (18, 255, 255)),    # hi-vis orange
        ]
        self._gloves_ranges = [
            ((18, 80, 120), (35, 255, 255)),    # yellow/orange work gloves
            ((0, 0, 30), (179, 60, 110)),        # dark grey/black gloves
        ]
        self._boots_ranges = [
            ((0, 0, 15), (179, 80, 90)),          # dark work boots
            ((18, 60, 100), (32, 255, 255)),     # tan/yellow work boots
        ]
        self._mask_ranges = [
            ((90, 20, 150), (140, 120, 255)),    # light blue/white surgical mask
            ((0, 0, 20), (179, 60, 100)),          # black mask
        ]

    @staticmethod
    def _mask_ratio(hsv_region, ranges):
        if hsv_region.size == 0:
            return 0.0
        total = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.inRange(hsv_region, np.array(lo), np.array(hi))
            total = cv2.bitwise_or(total, mask)
        return float(np.count_nonzero(total)) / float(total.size)

    @staticmethod
    def _ratio_to_confidence(ratio, present):
        # Maps a raw pixel ratio to a pseudo-confidence in [0.01, 0.99] for
        # display purposes only - not a calibrated probability.
        return min(0.99, 0.5 + ratio * 2) if present else max(0.01, 0.5 - ratio)

    def classify_raw(self, crop_bgr):
        """Returns {label: (value, confidence)} for every label in ALL_TRAINABLE_LABELS."""
        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return {label: (False, 0.0) for label in config.ALL_TRAINABLE_LABELS}

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)

        head_region = hsv[0:int(h * 0.22), :]
        mask_region = hsv[int(h * 0.11):int(h * 0.22), :]           # lower half of the head region
        torso_region = hsv[int(h * 0.22):int(h * 0.75), :]
        hand_region = np.concatenate([                              # both side strips, hands hang near the sides when standing
            hsv[int(h * 0.35):int(h * 0.65), 0:max(1, int(w * 0.18))],
            hsv[int(h * 0.35):int(h * 0.65), int(w * 0.82):w],
        ], axis=1) if w > 4 else hsv[0:0, 0:0]
        feet_region = hsv[int(h * 0.88):h, :]

        ratios = {
            "helmet": self._mask_ratio(head_region, self._helmet_ranges),
            "vest": self._mask_ratio(torso_region, self._vest_ranges),
            "gloves": self._mask_ratio(hand_region, self._gloves_ranges),
            "boots": self._mask_ratio(feet_region, self._boots_ranges),
            "mask": self._mask_ratio(mask_region, self._mask_ranges),
        }
        thresholds = {
            "helmet": config.HEURISTIC_HELMET_PIXEL_RATIO,
            "vest": config.HEURISTIC_VEST_PIXEL_RATIO,
            "gloves": config.HEURISTIC_GLOVES_PIXEL_RATIO,
            "boots": config.HEURISTIC_BOOTS_PIXEL_RATIO,
            "mask": config.HEURISTIC_MASK_PIXEL_RATIO,
        }

        out = {}
        for label in config.PPE_LABELS:
            present = ratios[label] >= thresholds[label]
            out[label] = (present, self._ratio_to_confidence(ratios[label], present))

        fallen = (w / float(h)) >= config.FALL_ASPECT_RATIO_THRESHOLD
        fallen_conf = min(0.99, (w / float(h)) / config.FALL_ASPECT_RATIO_THRESHOLD * 0.6) if fallen else 0.1
        out["fallen"] = (fallen, fallen_conf)
        return out

    def classify(self, crop_bgr):
        """Standalone use (e.g. if both ViT models are missing): full PPEResult."""
        raw = self.classify_raw(crop_bgr)
        values = {label: v for label, (v, _c) in raw.items()}
        confidences = {label: c for label, (_v, c) in raw.items()}
        return PPEResult(values, confidences)


class CombinedClassifier:
    """
    Composes the PPE model, the fall model, and the heuristic fallback into
    one classifier with the PPEResult interface everything downstream
    already expects. Each of the two ViT models is used when available;
    the heuristic fills in for whichever one (if any) isn't trained yet -
    so PPE detection can be running the real model while fall detection is
    still in Demo Mode, or vice versa.
    """

    def __init__(self, ppe_model, fall_model, heuristic):
        self.ppe_model = ppe_model
        self.fall_model = fall_model
        self.heuristic = heuristic

        parts = []
        parts.append(ppe_model.name if ppe_model else "heuristic (PPE items)")
        parts.append(fall_model.name if fall_model else "heuristic (fallen)")
        self.name = f"PPE: {parts[0]} | Fallen: {parts[1]}"

    def classify(self, crop_bgr):
        # Only run the heuristic if at least one real model is missing -
        # it's pure CPU-side colour math, cheap, but no point computing it
        # when both trained models are loaded.
        heuristic_raw = self.heuristic.classify_raw(crop_bgr) if not (self.ppe_model and self.fall_model) else None

        values, confidences = {}, {}
        ppe_raw = self.ppe_model.classify_raw(crop_bgr) if self.ppe_model else None
        for label in config.PPE_LABELS:
            v, c = (ppe_raw or heuristic_raw)[label]
            values[label], confidences[label] = v, c

        fall_raw = self.fall_model.classify_raw(crop_bgr) if self.fall_model else None
        for label in config.FALL_LABELS:
            v, c = (fall_raw or heuristic_raw)[label]
            values[label], confidences[label] = v, c

        return PPEResult(values, confidences)


def _try_load_vit(checkpoint_dir, labels, what):
    if not (os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir)):
        log(f"No fine-tuned checkpoint found at {checkpoint_dir} for {what}. Using heuristic. Run train_vit.py to train your own.", "warn")
        return None
    try:
        model = ViTLabelClassifier(checkpoint_dir, labels)
        log(f"Loaded {what} classifier: {model.name}")
        return model
    except Exception as e:
        log(f"Could not load fine-tuned checkpoint for {what} ({e}). Falling back to heuristic.", "warn")
        return None


def build_classifier():
    """
    Attempts to load each of the two fine-tuned ViT checkpoints (PPE items,
    fallen) independently. Either one missing/untrained falls back to the
    colour+geometry heuristic for just that piece.
    Returns: (classifier, is_demo_mode: bool) - is_demo_mode is True if
    either piece is running on the heuristic fallback.
    """
    ppe_model = _try_load_vit(config.VIT_PPE_CHECKPOINT_DIR, config.PPE_LABELS, "PPE")
    fall_model = _try_load_vit(config.VIT_FALL_CHECKPOINT_DIR, config.FALL_LABELS, "fallen")
    heuristic = HeuristicPPEClassifier()

    clf = CombinedClassifier(ppe_model, fall_model, heuristic)
    demo_mode = not (ppe_model and fall_model)
    log(f"Classifier ready: {clf.name}", "warn" if demo_mode else "info")
    return clf, demo_mode
