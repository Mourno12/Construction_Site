"""
drawing.py
Draws bounding boxes, worker ID / compliance labels, tracked-object markers,
and a small HUD stats strip onto frames before they're sent back to the
dashboard.
"""

import cv2
import config


def draw_worker_box(frame, bbox, track_id, ppe_result, compliance_rate=None):
    x1, y1, x2, y2 = bbox

    if ppe_result.fallen:
        color = config.COLOR_FALLEN
        status = "FALLEN"
    elif ppe_result.compliant:
        color = config.COLOR_SAFE
        status = "SAFE"
    else:
        color = config.COLOR_UNSAFE
        status = "VIOLATION"

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"ID {track_id}"
    if compliance_rate is not None:
        tag += f"  {compliance_rate:.0f}%"

    if ppe_result.fallen:
        detail = "WORKER DOWN"
    else:
        missing = [label.capitalize() for label in config.PPE_LABELS if not getattr(ppe_result, label)]
        detail = "All PPE OK" if not missing else "Missing: " + ", ".join(missing)

    label_lines = [f"{tag}  |  {status}", detail]
    line_h = 18
    box_w = max(cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)[0][0] for line in label_lines) + 12
    box_h = line_h * len(label_lines) + 6

    label_y1 = max(0, y1 - box_h)
    cv2.rectangle(frame, (x1, label_y1), (x1 + box_w, y1), color, -1)
    for i, line in enumerate(label_lines):
        ty = label_y1 + (i + 1) * line_h - 4
        cv2.putText(frame, line, (x1 + 6, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_object_box(frame, bbox, track_id, falling=False):
    """Marks a tracked non-person object (config.YOLO_OBJECT_CLASS_IDS),
    highlighting it distinctly while an object-fall event is active."""
    x1, y1, x2, y2 = bbox
    color = config.COLOR_OBJECT_FALL if falling else config.COLOR_OBJECT
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3 if falling else 2)

    tag = f"OBJ {track_id}" + ("  FALLING!" if falling else "")
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    label_y1 = max(0, y1 - th - 10)
    cv2.rectangle(frame, (x1, label_y1), (x1 + tw + 12, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 6, y1 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_hud(frame, stats, demo_mode=False):
    h, w = frame.shape[:2]
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    text = (f"Workers: {stats.get('totalWorkers', 0)}   "
            f"Safe: {stats.get('safeCount', 0)}   "
            f"Unsafe: {stats.get('unsafeCount', 0)}   "
            f"Fallen: {stats.get('fallenCount', 0)}   "
            f"Objects: {stats.get('trackedObjects', 0)}   "
            f"FPS: {stats.get('fps', 0):.1f}")
    cv2.putText(frame, text, (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                (255, 255, 255), 1, cv2.LINE_AA)

    if demo_mode:
        badge = "DEMO MODE"
        (tw, th), _ = cv2.getTextSize(badge, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (w - tw - 20, 4), (w - 4, bar_h - 4), (0, 145, 245), -1)
        cv2.putText(frame, badge, (w - tw - 12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (10, 10, 10), 2, cv2.LINE_AA)
    return frame
