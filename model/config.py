from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Union
import os


@dataclass
class DetectorConfig:
    # Model — YOLOv10 weights (NMS-free end-to-end detection)
    _BASE_PATH = os.path.abspath(__file__)
    model_path: str = os.path.join(os.path.dirname(_BASE_PATH), "..", "train14_ncnn_model_Fast")
    device: str = "cuda"                   # "cpu", "cuda", "cuda:0", "mps"

    # Inference thresholds
    confidence: float = 0.5              # Minimum confidence to keep a detection
    # Note: YOLOv10 is NMS-free — iou_threshold is not used during inference
    iou_threshold: float = 0.45

    # Target filter — empty list means detect all classes
    target_classes: List[int] = field(default_factory=list)

    # Input resolution fed to the model (None = model default)
    imgsz: Union[int,Tuple[int,int]] = (1280,720)

    # Video sources
    source: List[str] = field(default_factory=lambda: ["rtsp://10.0.0.100:8554/live"])

    # Display
    show: bool = True                   # Show live annotated frames
    window_name: str = "CVEngine"

    # Recording — set output_path to save annotated video
    output_path: Optional[str] = None
    fps: int = 30

    def set_model_path(self, value: str) -> None:
        self.model_path = value

    def set_device(self, value: str) -> None:
        self.device = value

    def set_confidence(self, value: float) -> None:
        self.confidence = float(value)

    def set_iou_threshold(self, value: float) -> None:
        self.iou_threshold = float(value)

    def set_target_classes(self, value: List[int]) -> None:
        self.target_classes = value

    def set_imgsz(self, value: Union[int, Tuple[int, int]]) -> None:
        self.imgsz = value

    def set_source(self, value: List[str]) -> None:
        self.source = value

    def set_show(self, value: bool) -> None:
        self.show = value

    def set_window_name(self, value: str) -> None:
        self.window_name = value

    def set_output_path(self, value: Optional[str]) -> None:
        self.output_path = value

    def set_fps(self, value: int) -> None:
        self.fps = int(value)
