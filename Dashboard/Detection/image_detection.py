#!/usr/bin/env python3
import cv2
import os
from tracking_detector import TrackingDetector


def process_image(input_path, output_path):

    if not os.path.exists(input_path):
        raise Exception("Image not found")

    image = cv2.imread(input_path)

    if image is None:
        raise Exception("Invalid image")

    detector = TrackingDetector()

    results = detector.process_frame(image)
    annotated = detector.draw_results(image, results)

    cv2.imwrite(output_path, annotated)

    return {
        "success": True,
        "output_path": output_path,
        "analysis": {
            "workers": len(results["workers"]),
            "safe": results["safe_count"],
            "unsafe": results["unsafe_count"],
            "compliance": results["compliance_rate"]
        }
    }