"""
tracker.py
Lightweight IOU-based multi-object tracker used ONLY as a fallback when
the real ByteTrack (shipped inside ultralytics) can't be used, e.g. because
torch/ultralytics aren't installed. Assigns persistent integer worker IDs
across frames using greedy IOU matching + a short "grace period" for
missed detections so an ID survives brief occlusion or the worker briefly
leaving frame, similar in spirit to ByteTrack's low-confidence recovery.
"""

import config


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / float(area_a + area_b - inter)


class Track:
    __slots__ = ("track_id", "bbox", "missed_frames")

    def __init__(self, track_id, bbox):
        self.track_id = track_id
        self.bbox = bbox
        self.missed_frames = 0


class SimpleIOUTracker:
    def __init__(self):
        self._next_id = 1
        self.tracks = {}  # track_id -> Track

    def update(self, detections):
        """
        detections: list of detector.Detection
        Returns: list of (track_id, Detection) in the same order as input.
        """
        unmatched_dets = list(range(len(detections)))
        matched_pairs = []

        for track_id, track in list(self.tracks.items()):
            best_iou, best_idx = 0.0, -1
            for di in unmatched_dets:
                score = iou(track.bbox, detections[di].bbox)
                if score > best_iou:
                    best_iou, best_idx = score, di
            if best_iou >= config.TRACKER_MIN_IOU:
                matched_pairs.append((track_id, best_idx))
                unmatched_dets.remove(best_idx)

        results = [None] * len(detections)
        matched_track_ids = set()
        for track_id, di in matched_pairs:
            self.tracks[track_id].bbox = detections[di].bbox
            self.tracks[track_id].missed_frames = 0
            results[di] = track_id
            matched_track_ids.add(track_id)

        # New tracks for unmatched detections
        for di in unmatched_dets:
            new_id = self._next_id
            self._next_id += 1
            self.tracks[new_id] = Track(new_id, detections[di].bbox)
            results[di] = new_id

        # Age out tracks that had no match this frame
        for track_id, track in list(self.tracks.items()):
            if track_id not in matched_track_ids and track_id not in [t for t, d in matched_pairs]:
                if track_id not in [results[i] for i in range(len(results)) if results[i] is not None]:
                    track.missed_frames += 1
                if track.missed_frames > config.TRACKER_MAX_MISSED_FRAMES:
                    del self.tracks[track_id]

        return [(results[i], detections[i]) for i in range(len(detections))]
