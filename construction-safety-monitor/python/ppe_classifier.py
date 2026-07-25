"""
ppe_classifier.py
Classifies a cropped worker image for PPE compliance: helmet (yes/no) and
hi-vis vest (yes/no).

Primary path : fine-tuned Vision Transformer (ViT-Base/16), trained with
               train_vit.py, loaded from config.VIT_CHECKPOINT_DIR.
               2 sigmoid outputs -> (helmet_prob, vest_prob).
Fallback path: colour-heuristic ("DEMO MODE") - looks for hard-hat-coloured
               pixels in the head region and hi-vis-coloured pixels in the
               torso region. No training/weights required, so the whole
               pipeline is runnable immediately after `git clone`.
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
    __slots__ = ("helmet", "vest", "helmet_conf", "vest_conf")

    def __init__(self, helmet, vest, helmet_conf, vest_conf):
        self.helmet = helmet
        self.vest = vest
        self.helmet_conf = helmet_conf
        self.vest_conf = vest_conf

    @property
    def compliant(self):
        return self.helmet and self.vest if config.REQUIRE_BOTH_FOR_COMPLIANCE else (self.helmet or self.vest)

    def to_dict(self):
        return {
            "helmet": self.helmet,
            "vest": self.vest,
            "helmetConfidence": round(self.helmet_conf, 3),
            "vestConfidence": round(self.vest_conf, 3),
            "compliant": self.compliant,
        }


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
            logits = self.model(**inputs).logits  # shape (1, 2): [helmet, vest]
            probs = self.torch.sigmoid(logits)[0].cpu().numpy()
        helmet_conf, vest_conf = float(probs[0]), float(probs[1])
        return PPEResult(
            helmet=helmet_conf >= config.HELMET_THRESHOLD,
            vest=vest_conf >= config.VEST_THRESHOLD,
            helmet_conf=helmet_conf,
            vest_conf=vest_conf,
        )


class HeuristicPPEClassifier:
    """
    DEMO MODE fallback - no trained weights required.
    Approximates helmet/vest detection using HSV colour masks:
      - Helmet colours: safety yellow, orange, white, red (top ~22% of the
        person crop = head region).
      - Vest colours: hi-vis yellow/lime or hi-vis orange, usually with
        reflective strips (mid torso region, ~22%-75% of crop height).
    This is intentionally simple so it runs everywhere; swap in the real
    ViT model for production-grade accuracy.
    """

    def __init__(self):
        self.name = "Colour-heuristic classifier (DEMO MODE - train/plug in ViT for real accuracy)"
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

    @staticmethod
    def _mask_ratio(hsv_region, ranges):
        if hsv_region.size == 0:
            return 0.0
        total = np.zeros(hsv_region.shape[:2], dtype=np.uint8)
        for lo, hi in ranges:
            mask = cv2.inRange(hsv_region, np.array(lo), np.array(hi))
            total = cv2.bitwise_or(total, mask)
        return float(np.count_nonzero(total)) / float(total.size)

    def classify(self, crop_bgr):
        h, w = crop_bgr.shape[:2]
        if h < 10 or w < 10:
            return PPEResult(False, False, 0.0, 0.0)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        head_region = hsv[0:int(h * 0.22), :]
        torso_region = hsv[int(h * 0.22):int(h * 0.75), :]

        helmet_ratio = self._mask_ratio(head_region, self._helmet_ranges)
        vest_ratio = self._mask_ratio(torso_region, self._vest_ranges)

        helmet = helmet_ratio >= config.HEURISTIC_HELMET_PIXEL_RATIO
        vest = vest_ratio >= config.HEURISTIC_VEST_PIXEL_RATIO

        # Map pixel ratio to a pseudo-confidence in [0.5, 0.99] for display
        helmet_conf = min(0.99, 0.5 + helmet_ratio * 2) if helmet else max(0.01, 0.5 - helmet_ratio)
        vest_conf = min(0.99, 0.5 + vest_ratio * 2) if vest else max(0.01, 0.5 - vest_ratio)

        return PPEResult(helmet, vest, helmet_conf, vest_conf)


def build_classifier():
    """
    Attempts to load the fine-tuned ViT checkpoint. Falls back to the
    colour heuristic if the checkpoint is missing or torch/transformers
    aren't installed.
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
