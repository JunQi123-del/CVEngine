from __future__ import annotations

from typing import Any

import cv2
from ultralytics import YOLO

from model.config import DetectorConfig


class YOLODetector:
    """
    Wraps an Ultralytics YOLO model for real-time target detection on live video.

    Usage:
        config = DetectorConfig(model_path="yolov8n.pt", source=0, confidence=0.5)
        detector = YOLODetector(config)
        detector.run()
    """

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self.model = YOLO(config.model_path)
        self.model.to(config.device)
        self._writer: cv2.VideoWriter | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open the video source and run detection until the user presses 'q'."""
        cap = self._open_capture()
        try:
            self._loop(cap)
        finally:
            cap.release()
            if self._writer:
                self._writer.release()
            cv2.destroyAllWindows()

    def predict_frame(self, frame):
        """
        Run inference on a single BGR numpy frame.

        Returns:
            results: ultralytics Results object
            annotated: BGR numpy array with bounding boxes drawn
        """
        results = self.model.predict(
            source=frame,
            conf=self.cfg.confidence,
            iou=self.cfg.iou_threshold,
            classes=self.cfg.target_classes or None,
            imgsz=self.cfg.imgsz,
            device=self.cfg.device,
            verbose=False,
        )
        annotated = results[0].plot()
        return results[0], annotated

    def predict_video(self, source: int | str) -> list[dict[str, Any]]:
        """
        Run YOLOv10 inference on every frame of a video source.

        Args:
            source: Video file path, RTSP/HTTP URL, or webcam index.

        Returns:
            List of per-frame dicts::

                [
                    {
                        "frame_index": 0,
                        "detections": [
                            {
                                "class_id": int,
                                "class_name": str,
                                "confidence": float,
                                "bbox": [x1, y1, x2, y2],   # absolute pixel coords
                            },
                            ...
                        ],
                    },
                    ...
                ]
        """
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {source!r}")

        frames: list[dict[str, Any]] = []
        frame_index = 0

        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                results, _ = self.predict_frame(frame)

                detections = [
                    {
                        "class_id": int(box.cls[0]),
                        "class_name": self.model.names[int(box.cls[0])],
                        "confidence": round(float(box.conf[0]), 4),
                        "bbox": [round(v, 2) for v in box.xyxy[0].tolist()],
                    }
                    for box in results.boxes
                ]

                frames.append({"frame_index": frame_index, "detections": detections})
                frame_index += 1
        finally:
            cap.release()

        return frames

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_capture(self) -> cv2.VideoCapture:
        cap = cv2.VideoCapture(self.cfg.source)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video source: {self.cfg.source!r}")
        return cap

    def _init_writer(self, frame) -> cv2.VideoWriter:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(self.cfg.output_path, fourcc, self.cfg.fps, (w, h))

    def _loop(self, cap: cv2.VideoCapture) -> None:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            _, annotated = self.predict_frame(frame)

            if self.cfg.output_path:
                if self._writer is None:
                    self._writer = self._init_writer(annotated)
                self._writer.write(annotated)

            if self.cfg.show:
                cv2.imshow(self.cfg.window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

