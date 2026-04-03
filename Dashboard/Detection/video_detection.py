#!/usr/bin/env python3
"""
video_detection.py - FIXED VERSION
Stable video processing with correct frame size + FFmpeg conversion
"""

import cv2
import os
import subprocess
from tracking_detector import TrackingDetector


def process_video(input_path, output_path):

    detector = TrackingDetector()

    temp_output = output_path.replace(".mp4", "_temp.avi")

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        raise Exception("Could not open input video")

    # ✅ FIXED FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25

    # 🔥 FIXED FRAME SIZE (IMPORTANT)
    width = 1280
    height = 720

    # ✅ FIXED WRITER (MATCHES RESIZE)
    writer = cv2.VideoWriter(
        temp_output,
        cv2.VideoWriter_fourcc(*'MJPG'),
        int(fps),
        (width, height)
    )

    print("🚀 Processing video with tracking...")

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            continue

        # 🔥 Resize BEFORE processing (VERY IMPORTANT)
        frame = cv2.resize(frame, (width, height))

        # Process detection
        results = detector.process_frame(frame)

        # Draw results
        annotated = detector.draw_results(frame, results)

        # ✅ WRITE SAFE FRAME
        writer.write(annotated)

        frame_count += 1

    cap.release()
    writer.release()

    print("🎥 Converting to MP4...")

    # ✅ FFmpeg conversion (SAFE)
    result = subprocess.run(
        [
            "ffmpeg", "-y",
            "-i", temp_output,
            "-vcodec", "libx264",
            "-pix_fmt", "yuv420p",
            output_path
        ],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print("❌ FFmpeg ERROR:", result.stderr)
        raise Exception("FFmpeg conversion failed")

    # Clean temp file
    if os.path.exists(temp_output):
        os.remove(temp_output)

    # ✅ RETURN DASHBOARD DATA
    worker_logs = detector.get_worker_logs()
    compliance_history = detector.get_compliance_history()

    avg_fps = (
        sum(detector.fps_history) / len(detector.fps_history)
        if detector.fps_history else 0
    )

    return {
        "success": True,
        "output_path": output_path,
        "tracking_data": {
            "total_workers": len(worker_logs),
            "worker_logs": worker_logs,
            "compliance_history": compliance_history,
            "average_fps": round(avg_fps, 2),
            "frames_processed": frame_count
        }
    }