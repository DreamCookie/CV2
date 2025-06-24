import cv2
import numpy as np
from typing import List, Tuple, Dict
from .base import Detector, BoundingBox

class ContourDetector(Detector):
    def __init__(self,
                 min_area: int = 1000,
                 blur_ksize: Tuple[int, int] = (5, 5),
                 threshold_params: Dict = None,
                 nms_iou_thresh: float = 0.3):
        """
        min_area минимальная площадь контура в пикселях
        blur_ksize размер ядра для сглаживания
        threshold_params параметры для cv2.threshold или cv2.adaptiveThreshold
        nms_iou_thresh порог для Non-Maximum Suppression
        """
        self.min_area = min_area
        self.blur_ksize = blur_ksize
        self.threshold_params = threshold_params or {
            'type': 'adaptive',      # 'adaptive' | 'otsu' | 'fixed'
            'max_value': 255,
            'method': 'gaussian',
            'block_size': 11,
            'C': 2,
            'thresh': 127            
        }
        self.nms_iou_thresh = nms_iou_thresh

    def detect(self, frame: np.ndarray) -> List[BoundingBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, self.blur_ksize, 0)

        # Бинаризация
        tp = self.threshold_params
        if tp['type'] == 'adaptive':
            mask = cv2.adaptiveThreshold(
                blur, tp['max_value'],
                cv2.ADAPTIVE_THRESH_MEAN_C if tp['method']=='mean' else cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                tp['block_size'], tp['C']
            )
        elif tp['type'] == 'otsu':
            _, mask = cv2.threshold(
                blur, 0, tp['max_value'],
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
        else:
            _, mask = cv2.threshold(
                blur, tp['thresh'], tp['max_value'],
                cv2.THRESH_BINARY_INV
            )

        # Поиск контуров
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes: List[BoundingBox] = []
        for cnt in cnts:
            if cv2.contourArea(cnt) < self.min_area:
                continue
            x, y, w, h = cv2.boundingRect(cnt)
            boxes.append(BoundingBox(x, y, x + w, y + h))

        # Non-Maximum Suppression
        return self._apply_nms(boxes, self.nms_iou_thresh)

    @staticmethod
    def _apply_nms(boxes: List[BoundingBox], iou_threshold: float) -> List[BoundingBox]:
        if not boxes:
            return []
        # преобразуем в массивы
        x1 = np.array([b.x_min for b in boxes])
        y1 = np.array([b.y_min for b in boxes])
        x2 = np.array([b.x_max for b in boxes])
        y2 = np.array([b.y_max for b in boxes])
        scores = np.array([b.score if b.score is not None else 1.0 for b in boxes])

        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            iou = inter / (areas[i] + areas[order[1:]] - inter)

            inds = np.where(iou <= iou_threshold)[0]
            order = order[inds + 1]

        return [boxes[idx] for idx in keep]
