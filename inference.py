from __future__ import annotations

import queue
import time
import threading
from typing import Any

import cv2
from ultralytics import YOLO

from model.config import DetectorConfig


class YOLODetector:
    """
    Wraps an Ultralytics YOLO model for real-time target detection on live video.

    Usage:
        config = DetectorConfig(model_path="yolov10n.pt", source=["rtsp://...", "rtsp://..."])
        detector = YOLODetector(config)
        detector.run()
    """

    def __init__(self, config: DetectorConfig):
        self.cfg = config
        self.model = YOLO(config.model_path)
        self.model.to(config.device)

        self.run()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> None:
        """Open all video sources and run detection until the user presses 'q'.

        Inference runs in per-source threads; display is handled on the main
        thread to satisfy OpenCV's GUI thread requirement.
        """
        stop = threading.Event()
        queues: dict[str, queue.Queue] = {src: queue.Queue(maxsize=2) for src in self.cfg.source}

        threads = [
            threading.Thread(target=self._infer_loop, args=(src, queues[src], stop), daemon=True)
            for src in self.cfg.source
        ]
        for t in threads:
            t.start()

        try:
            while not stop.is_set():
                for src, q in queues.items():
                    try:
                        annotated = q.get_nowait()
                    except queue.Empty:
                        continue
                    window_name = f"{self.cfg.window_name}-{src}"
                    if self.cfg.show:
                        cv2.imshow(window_name, annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    stop.set()
                    break
        finally:
            stop.set()
            for t in threads:
                t.join()
            if self.cfg.show:
                cv2.destroyAllWindows()

    def _predict_frame(self, frame):
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

    def _predict_video(self, sources: list[str]) -> dict[str, list[dict[str, Any]]]:
        """
        Run YOLOv10 inference on every frame of each video source in parallel.

        Args:
            sources: List of video file paths or RTSP/HTTP URLs.

        Returns:
            Dict mapping each source string to its list of per-frame dicts::

                {
                    "rtsp://...": [
                        {
                            "frame_index": 0,
                            "detections": [
                                {
                                    "class_id": int,
                                    "class_name": str,
                                    "confidence": float,
                                    "bbox": [x1, y1, x2, y2],
                                },
                                ...
                            ],
                        },
                        ...
                    ],
                    ...
                }
        """
        results: dict[str, list[dict[str, Any]]] = {}
        lock = threading.Lock()

        def _worker(src: str) -> None:
            frames = self._infer_source(src)
            with lock:
                results[src] = frames

        threads = [threading.Thread(target=_worker, args=(src,)) for src in sources]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        return results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _open_capture(self, source: str, retry_interval: float = 5.0) -> cv2.VideoCapture:
        """Open a VideoCapture, retrying indefinitely until the source becomes available."""
        while True:
            cap = cv2.VideoCapture(source)
            if cap.isOpened():
                return cap
            cap.release()
            print(f"[CVEngine] Waiting for source {source!r}, retrying in {retry_interval}s...")
            time.sleep(retry_interval)

    def _infer_loop(self, source: str, q: queue.Queue, stop: threading.Event) -> None:
        """Inference thread: capture + detect, push annotated frames into q."""
        writer: cv2.VideoWriter | None = None
        try:
            while not stop.is_set():
                cap = self._open_capture(source)
                try:
                    while not stop.is_set():
                        ok, frame = cap.read()
                        if not ok:
                            print(f"[CVEngine] Lost feed from {source!r}, reconnecting...")
                            break

                        _, annotated = self.predict_frame(frame)

                        if self.cfg.output_path:
                            if writer is None:
                                writer = self._init_writer(annotated)
                            writer.write(annotated)

                        try:
                            q.put_nowait(annotated)
                        except queue.Full:
                            pass
                finally:
                    cap.release()
        finally:
            if writer:
                writer.release()
            # if self.cfg.show:
            #     cv2.destroyWindow(window_name)

    def _infer_source(self, source: str) -> list[dict[str, Any]]:
        """Run inference on every frame of a single video source."""
        frames: list[dict[str, Any]] = []
        frame_index = 0
        cap = self._open_capture(source)
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

    def _init_writer(self, frame) -> cv2.VideoWriter:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(self.cfg.output_path, fourcc, self.cfg.fps, (w, h))
