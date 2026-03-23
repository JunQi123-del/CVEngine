
from model.config import DetectorConfig
from inference import YOLODetector

if __name__ == "__main__":
    config = YOLODetector(DetectorConfig())
