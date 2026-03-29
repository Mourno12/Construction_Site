#!/usr/bin/env python3
"""
video_detection.py - WITH TRACKING
Processes video with ByteTrack, FPS, Worker IDs, Compliance History
"""

import cv2
import os
import subprocess
from tracking_detector import TrackingDetector

def process_video(input_path, output_path):
    """Process video with full tracking"""
    
    # Initialize tracking detector
    detector = TrackingDetector()
    
    temp_output = output_path.replace(".mp4", "_temp.avi")
    
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        raise Exception("Could not open input video")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        fps = 25
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    writer = cv2.VideoWriter(temp_output, fourcc, int(fps), (width, height))
    
    print("Processing video with tracking...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process with tracking
        results = detector.process_frame(frame)
        
        # Draw results
        annotated = detector.draw_results(frame, results)
        
        writer.write(annotated)
    
    cap.release()
    writer.release()
    
    print("Converting to MP4...")
    
    # Convert to MP4
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
        print("FFmpeg ERROR:", result.stderr)
        raise Exception("FFmpeg conversion failed")
    
    os.remove(temp_output)
    
    # Get tracking data
    worker_logs = detector.get_worker_logs()
    compliance_history = detector.get_compliance_history()
    
    return {
        "success": True,
        "output_path": output_path,
        "tracking_data": {
            "total_workers": len(worker_logs),
            "worker_logs": worker_logs,
            "compliance_history": compliance_history,
            "average_fps": sum(detector.fps_history) / len(detector.fps_history) if detector.fps_history else 0
        }
    }