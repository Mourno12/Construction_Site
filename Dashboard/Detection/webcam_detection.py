#!/usr/bin/env python3
import cv2
from tracking_detector import TrackingDetector


def process_webcam():

    detector = TrackingDetector()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise Exception("Webcam not found")

    print("Starting webcam... Press 'q' to exit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = detector.process_frame(frame)
        annotated = detector.draw_results(frame, results)

        cv2.imshow("Construction Safety - Webcam", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    return {
        "success": True,
        "message": "Webcam session ended"
    }