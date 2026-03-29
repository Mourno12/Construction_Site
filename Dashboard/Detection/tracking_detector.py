import cv2
import time
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class TrackingDetector:
    def __init__(self, model_path="best.pt"):
        # Use CUSTOM MODEL (helmet + vest)
        self.model = YOLO(model_path)

        self.tracker = DeepSort(max_age=30)

        self.last_time = time.time()
        self.current_fps = 0
        self.fps_history = deque(maxlen=30)

        self.worker_logs = {}

    # ================= FPS =================
    def update_fps(self):
        now = time.time()
        fps = 1 / max((now - self.last_time), 1e-6)
        self.last_time = now
        self.current_fps = fps
        self.fps_history.append(fps)
        return fps

    # ================= IOU =================
    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)

    # ================= PROCESS =================
    def process_frame(self, frame):
        results = self.model(frame)[0]

        persons = []
        helmets = []
        vests = []

        # Collect detections
        for box in results.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 0:  # person
                persons.append([x1, y1, x2, y2])
            elif cls == 1:  # helmet
                helmets.append([x1, y1, x2, y2])
            elif cls == 2:  # vest
                vests.append([x1, y1, x2, y2])

        # Track persons
        detections = [
            ([p[0], p[1], p[2]-p[0], p[3]-p[1]], 0.9, "person")
            for p in persons
        ]

        tracks = self.tracker.update_tracks(detections, frame=frame)

        output = []
        safe_count = 0
        unsafe_count = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            person_box = [l, t, r, b]

            has_helmet = any(self.iou(person_box, h) > 0.2 for h in helmets)
            has_vest = any(self.iou(person_box, v) > 0.2 for v in vests)

            compliant = has_helmet and has_vest

            if compliant:
                safe_count += 1
            else:
                unsafe_count += 1

            output.append({
                "id": tid,
                "bbox": person_box,
                "helmet": has_helmet,
                "vest": has_vest,
                "compliant": compliant
            })

            # log worker
            if tid not in self.worker_logs:
                self.worker_logs[tid] = {
                    "worker_id": tid,
                    "total_frames": 0,
                    "safe_frames": 0,
                    "no_helmet_frames": 0,
                    "no_vest_frames": 0
                }

            log = self.worker_logs[tid]
            log["total_frames"] += 1

            if compliant:
                log["safe_frames"] += 1
            if not has_helmet:
                log["no_helmet_frames"] += 1
            if not has_vest:
                log["no_vest_frames"] += 1

        self.update_fps()

        total = safe_count + unsafe_count
        compliance_rate = (safe_count / total * 100) if total else 0

        return {
            "workers": output,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "compliance_rate": compliance_rate
        }

    # ================= DRAW =================
    def draw_results(self, frame, results):
        for w in results["workers"]:
            x1, y1, x2, y2 = w["bbox"]

            color = (0, 255, 0) if w["compliant"] else (0, 0, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            label = f"ID {w['id']} | H:{int(w['helmet'])} V:{int(w['vest'])}"
            cv2.putText(frame, label, (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"FPS: {self.current_fps:.2f}",
                    (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2)

        return frame

    # ================= EXPORT =================
    def get_worker_logs(self):
        logs = []
        for w in self.worker_logs.values():
            total = w["total_frames"]
            comp = (w["safe_frames"] / total * 100) if total else 0
            w["compliance_rate"] = comp
            logs.append(w)
        return logs

    def get_compliance_history(self):
        return []