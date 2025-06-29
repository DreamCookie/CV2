import cv2
import numpy as np
from typing import List, Optional
from ultralytics import YOLO
from .base import Detector, BoundingBox

class NNDetector(Detector):
    def __init__(self, model_path: str, conf_thresh: float = 0.5):
        self.model = YOLO(model_path)
        self.conf_thresh = conf_thresh

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        results = self.model(frame)[0]
        boxes: List[BoundingBox] = []

        for box in results.boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            score = float(box.conf[0].cpu().numpy())
            if score < self.conf_thresh:
                continue
            boxes.append(BoundingBox(x1, y1, x2, y2, score))

        return boxes
