import cv2
import time
import os
from collections import deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort


class TrackingDetector:
    def __init__(self, model_path=None):
        if model_path is None:
            model_path = os.path.join(os.path.dirname(__file__), "best.pt")

        print("🔥 Loading model from:", model_path)
        self.model = YOLO(model_path)

        self.tracker = DeepSort(max_age=30)

        self.last_time = time.time()
        self.current_fps = 0
        self.fps_history = deque(maxlen=30)

        self.memory = {}
        self.worker_logs = {}
        self.compliance_history = []

    def update_fps(self):
        now = time.time()
        fps = 1 / max((now - self.last_time), 1e-6)
        self.last_time = now
        self.current_fps = fps
        self.fps_history.append(fps)

    def iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        inter = max(0, xB - xA) * max(0, yB - yA)
        areaA = (boxA[2]-boxA[0])*(boxA[3]-boxA[1])
        areaB = (boxB[2]-boxB[0])*(boxB[3]-boxB[1])

        return inter / (areaA + areaB - inter + 1e-6)

    def process_frame(self, frame):
        results = self.model(frame, conf=0.02, imgsz=1280, augment=True, verbose=False)[0]

        persons, helmets, vests, gloves, boots = [], [], [], [], []

        # MAIN DETECTION
        for box in results.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if cls == 3:
                persons.append([x1, y1, x2, y2])
            elif cls == 2:
                helmets.append([x1, y1, x2, y2])
            elif cls == 4:
                vests.append([x1, y1, x2, y2])
            elif cls == 1:
                gloves.append([x1, y1, x2, y2])
            elif cls == 0:
                boots.append([x1, y1, x2, y2])

        # 🔥 SMALL OBJECT BOOST (RIGHT SIDE ZOOM)
        h, w = frame.shape[:2]
        crop = frame[:, int(w * 0.6):]

        results_zoom = self.model(crop, conf=0.02, imgsz=1280, verbose=False)[0]

        for box in results_zoom.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            x1 += int(w * 0.6)
            x2 += int(w * 0.6)

            if cls == 3:
                persons.append([x1, y1, x2, y2])
            elif cls == 2:
                helmets.append([x1, y1, x2, y2])
            elif cls == 4:
                vests.append([x1, y1, x2, y2])

        if len(persons) == 0:
            return {
                "workers": [],
                "safe_count": 0,
                "unsafe_count": 0,
                "compliance_rate": 0
            }

        detections = []
        for p in persons:
            x1, y1, x2, y2 = p
            detections.append(([x1, y1, x2-x1, y2-y1], 0.9, "person"))

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

            has_helmet = any(self.iou(person_box, h) > 0.05 for h in helmets)
            has_vest   = any(self.iou(person_box, v) > 0.05 for v in vests)
            has_gloves = any(self.iou(person_box, g) > 0.05 for g in gloves)
            has_boots  = any(self.iou(person_box, b) > 0.05 for b in boots)

            if tid not in self.memory:
                self.memory[tid] = {"helmet": [], "vest": [], "gloves": [], "boots": []}

            for k, val in zip(["helmet","vest","gloves","boots"], [has_helmet,has_vest,has_gloves,has_boots]):
                self.memory[tid][k].append(val)
                self.memory[tid][k] = self.memory[tid][k][-5:]

            has_helmet = sum(self.memory[tid]["helmet"]) >= 3
            has_vest   = sum(self.memory[tid]["vest"]) >= 3
            has_gloves = sum(self.memory[tid]["gloves"]) >= 3
            has_boots  = sum(self.memory[tid]["boots"]) >= 3

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
                "gloves": has_gloves,
                "boots": has_boots,
                "compliant": compliant
            })

            if tid not in self.worker_logs:
                self.worker_logs[tid] = {"worker_id": tid, "total_frames": 0, "safe_frames": 0}

            self.worker_logs[tid]["total_frames"] += 1
            if compliant:
                self.worker_logs[tid]["safe_frames"] += 1

        self.update_fps()

        total = safe_count + unsafe_count
        compliance_rate = (safe_count / total * 100) if total else 0

        return {
            "workers": output,
            "safe_count": safe_count,
            "unsafe_count": unsafe_count,
            "compliance_rate": compliance_rate
        }

    def draw_results(self, frame, results):
        for w in results["workers"]:
            x1, y1, x2, y2 = w["bbox"]
            color = (0,255,0) if w["compliant"] else (0,0,255)

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            label = f"ID {w['id']} H:{int(w['helmet'])} V:{int(w['vest'])}"
            cv2.putText(frame, label, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return frame

    def get_worker_logs(self):
        logs = []
        for w in self.worker_logs.values():
            total = w["total_frames"]
            compliance = (w["safe_frames"] / total * 100) if total else 0
            logs.append({"worker_id": w["worker_id"], "compliance_rate": round(compliance,2)})
        return logs

    def get_compliance_history(self):
        return self.compliance_history