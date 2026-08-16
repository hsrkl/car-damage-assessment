import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO


class YoloLoader:
    def __init__(self, weights_path, confidence=0.2, imgsz=640):
        if not Path(weights_path).is_file():
            raise FileNotFoundError(f"Model weights not found at {weights_path}")
        self.confidence = confidence
        self.imgsz = imgsz
        self.model = YOLO(weights_path)

    @staticmethod
    def _read_source(source):
        if isinstance(source, str):
            img = cv2.imread(source)
            if img is None:
                raise FileNotFoundError(f"Could not read image at {source}")
            return img
        if isinstance(source, np.ndarray):
            return source
        raise TypeError("source must be a file path string or numpy array")

    def _to_detections(self, results):
        detections = []
        if not results.boxes:
            return detections
        for i, box in enumerate(results.boxes):
            polygon = []
            if results.masks:
                try:
                    polygon = results.masks.xyn[i].tolist()
                except (IndexError, AttributeError):
                    pass
            detections.append({
                "class": results.names[int(box.cls[0])],
                "conf": float(box.conf[0]),
                "polygon": polygon,
            })
        return detections

    def process(self, source):
        img = self._read_source(source)
        results = self.model(img, conf=self.confidence, verbose=False, imgsz=self.imgsz)[0]
        return {"detected": self._to_detections(results)}
