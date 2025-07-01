import cv2
import numpy as np
from typing import List, Tuple
from .base import Detector, BoundingBox

class ContourDetector(Detector):
    """
    Классический детектор grayscale → blur → threshold → findContours → NMS.
    """
    def __init__(self, min_area: float, blur_ksize: Tuple[int,int], threshold_params: dict, nms_iou_thresh: float):
        self.min_area = min_area
        self.blur_ksize = blur_ksize
        self.threshold_params = threshold_params
        self.nms_iou_thresh = nms_iou_thresh

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_ksize, 0)

        # бинаризация
        tp = self.threshold_params
        if tp['type'] == 'adaptive':
            mask = cv2.adaptiveThreshold(
                blur, tp['max_value'],
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C if tp['method']=='gaussian' else cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, tp['block_size'], tp['C']
            )
        elif tp['type'] == 'otsu':
            _, mask = cv2.threshold(blur, 0, tp['max_value'], cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
        else:
            _, mask = cv2.threshold(blur, tp['thresh'], tp['max_value'], cv2.THRESH_BINARY_INV)

        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x,y,w,h = cv2.boundingRect(cnt)
            boxes.append(BoundingBox(x, y, x+w, y+h, contour=cnt))

        return boxes
