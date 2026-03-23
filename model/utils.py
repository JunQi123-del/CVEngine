from __future__ import annotations

import cv2
import numpy as np


def draw_detections(frame: np.ndarray, results) -> np.ndarray:
    """
    Draw bounding boxes and labels on a frame from an ultralytics Results object.
    Returns a copy of the frame with annotations.
    """
    annotated = frame.copy()
    boxes = results.boxes

    if boxes is None or len(boxes) == 0:
        return annotated

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        cls_id = int(box.cls[0])
        label = results.names.get(cls_id, str(cls_id))

        color = _class_color(cls_id)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        text = f"{label} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(
            annotated, text, (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA,
        )

    return annotated


def _class_color(cls_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color per class ID."""
    np.random.seed(cls_id)
    return tuple(int(c) for c in np.random.randint(100, 255, 3))
