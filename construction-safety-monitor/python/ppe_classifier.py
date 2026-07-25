"""
ppe_classifier.py
Classifies a cropped worker image for PPE compliance (helmet, vest, gloves,
boots, mask) and worker state (fallen), one boolean + confidence per label
in config.ALL_TRAINABLE_LABELS.

Primary path : fine-tuned Vision Transformer (ViT-Base/16), trained with
               train_vit.py, loaded from config.VIT_CHECKPOINT_DIR.
               One sigmoid output per label -> {label: probability}.
Fallback path: colour + geometry heuristic ("DEMO MODE") - looks for
               PPE-item-coloured pixels in the relevant crop region for each
               PPE label, and flags "fallen" from the crop's aspect ratio.
               No training/weights required, so the whole pipeline is
               runnable immediately after `git clone`.
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


class ViTPPEClassifier:
    """Fine-tuned Vision Transformer multi-label classifier."""

    def __init__(self, checkpoint_dir=config.VIT_CHECKPOINT_DIR):
        import torch
        from transformers import ViTForImageClassification, ViTImageProcessor

        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = ViTImageProcessor.from_pretrained(checkpoint_dir)
        self.model = ViTForImageClassification.from_pretrained(checkpoint_dir)
        self.model.to(self.device)
        self.model.eval()
        self.name = f"Fine-tuned ViT ({checkpoint_dir}) on {self.device}"

    def classify(self, crop_bgr):
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits  # shape (1, len(ALL_TRAINABLE_LABELS))
            probs = self.torch.sigmoid(logits)[0].cpu().numpy()

        values, confidences = {}, {}
        for label, prob in zip(config.ALL_TRAINABLE_LABELS, probs):
            prob = float(prob)
            confidences[label] = prob
            values[label] = prob >= config.LABEL_THRESHOLDS[label]
        return PPEResult(values, confidences)


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

    def classify(self, crop_bgr):
        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            values = {label: False for label in config.ALL_TRAINABLE_LABELS}
            confidences = {label: 0.0 for label in config.ALL_TRAINABLE_LABELS}
            return PPEResult(values, confidences)

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

        values, confidences = {}, {}
        for label in config.PPE_LABELS:
            present = ratios[label] >= thresholds[label]
            values[label] = present
            confidences[label] = self._ratio_to_confidence(ratios[label], present)

        fallen = (w / float(h)) >= config.FALL_ASPECT_RATIO_THRESHOLD
        values["fallen"] = fallen
        confidences["fallen"] = min(0.99, (w / float(h)) / config.FALL_ASPECT_RATIO_THRESHOLD * 0.6) if fallen else 0.1

        return PPEResult(values, confidences)


def build_classifier():
    """
    Attempts to load the fine-tuned ViT checkpoint. Falls back to the
    colour+geometry heuristic if the checkpoint is missing or torch/
    transformers aren't installed.
    Returns: (classifier, is_demo_mode: bool)
    """
    if os.path.isdir(config.VIT_CHECKPOINT_DIR) and os.listdir(config.VIT_CHECKPOINT_DIR):
        try:
            clf = ViTPPEClassifier()
            log(f"Loaded PPE classifier: {clf.name}")
            return clf, False
        except Exception as e:
            log(f"Could not load fine-tuned ViT checkpoint ({e}). Falling back to heuristic classifier.", "warn")
    else:
        log(
            f"No fine-tuned ViT checkpoint found at {config.VIT_CHECKPOINT_DIR}. "
            "Using heuristic classifier. Run train_vit.py to train your own.",
            "warn",
        )
    clf = HeuristicPPEClassifier()
    log(f"Loaded PPE classifier: {clf.name}", "warn")
    return clf, True
