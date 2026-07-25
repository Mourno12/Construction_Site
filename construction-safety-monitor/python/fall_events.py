"""
fall_events.py
Tracks vertical motion of non-person tracked objects (see
config.YOLO_OBJECT_CLASS_IDS) across frames to flag "object fall" events.

There's no single-frame "does this look like a falling object" model to
train - falling is a property of motion over time, not of one static crop -
so this is a small physics-style rule sitting on top of the (trainable)
object detector + tracker: if a tracked object's bounding-box centre keeps
moving down fast enough for long enough, it's falling. The debounce
(streak + alerted flag) mirrors the same pattern used for PPE violations in
inference_worker.py, so a sustained fall reports once, not once per frame.
"""

import config


class ObjectFallTracker:
    def __init__(self):
        self.history = {}  # track_id -> {"last_y": float, "streak": int, "alerted": bool}

    def update(self, track_id, bbox):
        """
        bbox: (x1, y1, x2, y2) for this track in the current frame.
        Returns (is_falling: bool, is_new_event: bool) - is_new_event is
        True only on the frame the sustained-fall streak first crosses the
        threshold, so callers can emit exactly one alert per fall.
        """
        x1, y1, x2, y2 = bbox
        center_y = (y1 + y2) / 2.0

        state = self.history.get(track_id)
        if state is None:
            self.history[track_id] = {"last_y": center_y, "streak": 0, "alerted": False}
            return False, False

        downward_px = center_y - state["last_y"]
        state["last_y"] = center_y

        if downward_px >= config.OBJECT_FALL_MIN_DOWNWARD_PX_PER_FRAME:
            state["streak"] += 1
        else:
            state["streak"] = 0
            state["alerted"] = False

        is_falling = state["streak"] >= config.OBJECT_FALL_STREAK_FRAMES
        is_new_event = is_falling and not state["alerted"]
        if is_new_event:
            state["alerted"] = True

        return is_falling, is_new_event

    def prune(self, active_track_ids):
        """Drop history for tracks that no longer exist (left frame, track lost)."""
        stale = [tid for tid in self.history if tid not in active_track_ids]
        for tid in stale:
            del self.history[tid]
