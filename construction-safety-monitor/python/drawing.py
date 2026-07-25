"""
drawing.py
Draws bounding boxes, worker ID / compliance labels, and a small HUD stats
strip onto frames before they're sent back to the dashboard.
"""

import cv2
import config


def draw_worker_box(frame, bbox, track_id, ppe_result, compliance_rate=None):
    x1, y1, x2, y2 = bbox
    color = config.COLOR_SAFE if ppe_result.compliant else config.COLOR_UNSAFE

    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    tag = f"ID {track_id}"
    if compliance_rate is not None:
        tag += f"  {compliance_rate:.0f}%"
    status = "SAFE" if ppe_result.compliant else "VIOLATION"
    helmet_tag = "Helmet OK" if ppe_result.helmet else "No Helmet"
    vest_tag = "Vest OK" if ppe_result.vest else "No Vest"

    label_lines = [f"{tag}  |  {status}", f"{helmet_tag}  /  {vest_tag}"]
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


def draw_hud(frame, stats, demo_mode=False):
    h, w = frame.shape[:2]
    bar_h = 34
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (18, 18, 18), -1)
    frame[:] = cv2.addWeighted(overlay, 0.65, frame, 0.35, 0)

    text = (f"Workers: {stats.get('totalWorkers', 0)}   "
            f"Safe: {stats.get('safeCount', 0)}   "
            f"Unsafe: {stats.get('unsafeCount', 0)}   "
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
