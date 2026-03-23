from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DetectorConfig:
    # Model — YOLOv10 weights (NMS-free end-to-end detection)
    model_path: str = "yolov10n.pt"      # yolov10n/s/m/b/l/x.pt
    device: str = "cuda"                   # "cpu", "cuda", "cuda:0", "mps"

    # Inference thresholds
    confidence: float = 0.5              # Minimum confidence to keep a detection
    # Note: YOLOv10 is NMS-free — iou_threshold is not used during inference
    iou_threshold: float = 0.45

    # Target filter — empty list means detect all classes
    target_classes: List[int] = field(default_factory=list)

    # Input resolution fed to the model (None = model default)
    imgsz: Union[int,Tuple[int,int]] = (720,1280)

    # Video source
    source: int | str = "rtsp://10.0.0.100:8554/live"               # 0 = default webcam, or a video file path / RTSP URL

    # Display
    show: bool = True                   # Show live annotated frames
    window_name: str = "CVEngine"

    # Recording — set output_path to save annotated video
    output_path: Optional[str] = None
    fps: int = 30

    
