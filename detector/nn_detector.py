from ultralytics import YOLO
import numpy as np
from typing import List
from .base import Detector, BoundingBox

class NNDetector(Detector):
    def __init__(self, model_path: str, conf_thresh: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        results = self.model.predict(frame, conf=self.conf_thresh, verbose=False)
        boxes = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append(BoundingBox(int(x1), int(y1), int(x2), int(y2)))
        return boxes