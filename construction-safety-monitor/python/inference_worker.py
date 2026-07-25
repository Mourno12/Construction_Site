"""
inference_worker.py
Long-lived worker process spawned once by the Node.js backend
(server/utils/pythonBridge.js). Speaks newline-delimited JSON (NDJSON):

  STDIN  (commands from Node)
    {"type": "frame", "sessionId": "...", "mode": "webcam"|"image", "image": "<base64>"}
    {"type": "start_video", "sessionId": "...", "path": "/abs/path/video.mp4"}
    {"type": "stop", "sessionId": "..."}
    {"type": "reset_session", "sessionId": "..."}

  STDOUT (events to Node -> forwarded to the browser over WebSocket)
    {"type": "ready", "detector": "...", "classifier": "...", "demoMode": bool}
    {"type": "result", "sessionId": "...", "mode": "...", "frameIndex": N,
     "detections": [...], "objects": [...], "stats": {...},
     "annotated": "data:image/jpeg;base64,..."}
    {"type": "video_done", "sessionId": "..."}
    {"type": "error", "sessionId": "...", "message": "..."}

  Diagnostic logs are written to STDERR (never STDOUT) so they never corrupt
  the NDJSON stream: {"type": "log", "level": "...", "message": "..."}

  Each entry in `stats.newAlerts` is {"trackId": N, "kind": "ppe"|"fall"|"object_fall"}:
    - "ppe"         - a worker missing required PPE for config.ALERT_STREAK_FRAMES frames
    - "fall"        - a worker detected as fallen for config.ALERT_STREAK_FRAMES frames
    - "object_fall" - a tracked object (config.YOLO_OBJECT_CLASS_IDS) falling
                       for config.OBJECT_FALL_STREAK_FRAMES frames
"""

import sys
import json
import time
import base64
import threading

import numpy as np
import cv2

import config
from detector import build_detector
from tracker import SimpleIOUTracker
from ppe_classifier import build_classifier
from fall_events import ObjectFallTracker
import drawing


def emit(obj):
    sys.stdout.write(json.dumps(obj) + "\n")
    sys.stdout.flush()


def log(message, level="info"):
    sys.stderr.write(json.dumps({"type": "log", "level": level, "message": message}) + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# Load models once at startup
# ---------------------------------------------------------------------------
DETECTOR, HAS_BUILTIN_TRACKING = build_detector()
CLASSIFIER, DEMO_MODE_CLASSIFIER = build_classifier()
DEMO_MODE = DEMO_MODE_CLASSIFIER or not HAS_BUILTIN_TRACKING

emit({
    "type": "ready",
    "detector": getattr(DETECTOR, "name", "unknown"),
    "classifier": getattr(CLASSIFIER, "name", "unknown"),
    "builtinTracking": HAS_BUILTIN_TRACKING,
    "demoMode": DEMO_MODE,
})


class SessionState:
    def __init__(self):
        self.tracker = SimpleIOUTracker() if not HAS_BUILTIN_TRACKING else None
        self.history = {}          # track_id -> dict (person tracks only)
        self.object_fall_tracker = ObjectFallTracker()
        self.frame_index = 0
        self.frame_times = []      # rolling timestamps for FPS calc
        self.stop_event = threading.Event()

    def rolling_fps(self):
        now = time.time()
        self.frame_times = [t for t in self.frame_times if now - t < 2.0]
        self.frame_times.append(now)
        if len(self.frame_times) < 2:
            return 0.0
        span = self.frame_times[-1] - self.frame_times[0]
        return (len(self.frame_times) - 1) / span if span > 0 else 0.0


SESSIONS = {}
SESSIONS_LOCK = threading.Lock()


def get_session(session_id):
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            SESSIONS[session_id] = SessionState()
        return SESSIONS[session_id]


def decode_base64_image(data_url):
    if "," in data_url and data_url.strip().startswith("data:"):
        data_url = data_url.split(",", 1)[1]
    raw = base64.b64decode(data_url)
    arr = np.frombuffer(raw, dtype=np.uint8)
    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    return frame


def encode_frame_to_data_url(frame):
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), config.JPEG_QUALITY])
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/jpeg;base64,{b64}"


def resize_if_needed(frame):
    h, w = frame.shape[:2]
    if w > config.MAX_FRAME_WIDTH:
        scale = config.MAX_FRAME_WIDTH / float(w)
        frame = cv2.resize(frame, (config.MAX_FRAME_WIDTH, int(h * scale)))
    return frame


def run_detection_and_tracking(frame, session):
    if HAS_BUILTIN_TRACKING:
        return DETECTOR.detect_and_track(frame)
    raw_detections = DETECTOR.detect(frame)
    return session.tracker.update(raw_detections)


def process_frame(frame, session, mode):
    frame = resize_if_needed(frame)
    tracked = run_detection_and_tracking(frame, session)
    frame_h, frame_w = frame.shape[:2]

    detections_payload = []
    objects_payload = []
    safe_count = 0
    unsafe_count = 0
    fallen_count = 0
    new_alerts = []
    active_object_ids = set()

    for track_id, det in tracked:
        x1, y1, x2, y2 = det.bbox
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame_w, x2), min(frame_h, y2)
        if x2 <= x1 or y2 <= y1:
            continue

        if not det.is_person:
            # Tracked non-person object (config.YOLO_OBJECT_CLASS_IDS) - run
            # the velocity-based fall check instead of PPE classification.
            active_object_ids.add(track_id)
            is_falling, is_new_event = session.object_fall_tracker.update(track_id, (x1, y1, x2, y2))
            if is_new_event:
                new_alerts.append({"trackId": track_id, "kind": "object_fall"})
            drawing.draw_object_box(frame, (x1, y1, x2, y2), track_id, falling=is_falling)
            objects_payload.append({
                "trackId": track_id,
                "bbox": [x1, y1, x2, y2],
                "confidence": round(det.confidence, 3),
                "falling": is_falling,
                "isNewAlert": is_new_event,
            })
            continue

        crop = frame[y1:y2, x1:x2]
        ppe = CLASSIFIER.classify(crop)

        h = session.history.setdefault(track_id, {
            "firstSeen": time.time(),
            "framesSeen": 0,
            "framesCompliant": 0,
            "badStreak": 0,
            "alerted": False,
        })
        h["lastSeen"] = time.time()
        h["framesSeen"] += 1
        if ppe.fallen:
            fallen_count += 1
        if ppe.compliant:
            h["framesCompliant"] += 1
            h["badStreak"] = 0
            h["alerted"] = False
            safe_count += 1
        else:
            h["badStreak"] += 1
            unsafe_count += 1
            if h["badStreak"] >= config.ALERT_STREAK_FRAMES and not h["alerted"]:
                h["alerted"] = True
                new_alerts.append({"trackId": track_id, "kind": "fall" if ppe.fallen else "ppe"})

        compliance_rate = 100.0 * h["framesCompliant"] / h["framesSeen"] if h["framesSeen"] else 0.0

        drawing.draw_worker_box(frame, (x1, y1, x2, y2), track_id, ppe, compliance_rate)

        new_alert_track_ids = {a["trackId"] for a in new_alerts}
        detections_payload.append({
            "trackId": track_id,
            "bbox": [x1, y1, x2, y2],
            "confidence": round(det.confidence, 3),
            **ppe.to_dict(),
            "complianceRate": round(compliance_rate, 1),
            "framesSeen": h["framesSeen"],
            "firstSeen": h["firstSeen"],
            "lastSeen": h["lastSeen"],
            "isNewAlert": track_id in new_alert_track_ids,
        })

    session.object_fall_tracker.prune(active_object_ids)

    session.frame_index += 1
    fps = session.rolling_fps()
    total_workers = len(detections_payload)
    stats = {
        "frameIndex": session.frame_index,
        "totalWorkers": total_workers,
        "safeCount": safe_count,
        "unsafeCount": unsafe_count,
        "fallenCount": fallen_count,
        "trackedObjects": len(objects_payload),
        "complianceRatePct": round(100.0 * safe_count / total_workers, 1) if total_workers else 100.0,
        "fps": round(fps, 1),
        "demoMode": DEMO_MODE,
        "newAlerts": new_alerts,
    }

    drawing.draw_hud(frame, stats, demo_mode=DEMO_MODE)
    return frame, detections_payload, objects_payload, stats


def handle_frame_command(cmd):
    session_id = cmd["sessionId"]
    session = get_session(session_id)
    try:
        frame = decode_base64_image(cmd["image"])
        if frame is None:
            raise ValueError("Could not decode image data")
        annotated, detections, objects, stats = process_frame(frame, session, cmd.get("mode", "webcam"))
        emit({
            "type": "result",
            "sessionId": session_id,
            "mode": cmd.get("mode", "webcam"),
            "detections": detections,
            "objects": objects,
            "stats": stats,
            "annotated": encode_frame_to_data_url(annotated),
        })
    except Exception as e:
        emit({"type": "error", "sessionId": session_id, "message": str(e)})


def handle_start_video(cmd):
    session_id = cmd["sessionId"]
    path = cmd["path"]
    session = get_session(session_id)
    session.stop_event.clear()

    def worker():
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            emit({"type": "error", "sessionId": session_id, "message": f"Could not open video: {path}"})
            return
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        frame_interval = 1.0 / min(source_fps, config.TARGET_FPS)
        try:
            while not session.stop_event.is_set():
                ok, frame = cap.read()
                if not ok:
                    break
                t0 = time.time()
                annotated, detections, objects, stats = process_frame(frame, session, "video")
                emit({
                    "type": "result",
                    "sessionId": session_id,
                    "mode": "video",
                    "detections": detections,
                    "objects": objects,
                    "stats": stats,
                    "annotated": encode_frame_to_data_url(annotated),
                })
                elapsed = time.time() - t0
                sleep_for = frame_interval - elapsed
                if sleep_for > 0:
                    time.sleep(sleep_for)
        finally:
            cap.release()
            emit({"type": "video_done", "sessionId": session_id})

    t = threading.Thread(target=worker, daemon=True)
    t.start()


def handle_stop(cmd):
    session_id = cmd["sessionId"]
    session = get_session(session_id)
    session.stop_event.set()


def handle_reset_session(cmd):
    session_id = cmd["sessionId"]
    with SESSIONS_LOCK:
        SESSIONS.pop(session_id, None)


COMMANDS = {
    "frame": handle_frame_command,
    "start_video": handle_start_video,
    "stop": handle_stop,
    "reset_session": handle_reset_session,
}


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            cmd = json.loads(line)
        except json.JSONDecodeError:
            log(f"Received malformed JSON line: {line[:200]}", "error")
            continue
        handler = COMMANDS.get(cmd.get("type"))
        if handler is None:
            log(f"Unknown command type: {cmd.get('type')}", "warn")
            continue
        try:
            handler(cmd)
        except Exception as e:
            emit({"type": "error", "sessionId": cmd.get("sessionId"), "message": str(e)})


if __name__ == "__main__":
    main()
