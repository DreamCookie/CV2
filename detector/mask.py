import cv2
import numpy as np

class BackgroundMasker:
    """
    Статическое фоновое вычитание по заранее снятому фону + применение ROI-масок
    """
    def __init__(self, background: np.ndarray = None, roi_mask: np.ndarray = None, thresh: int = 30):
        self.background = background
        self.roi_mask = roi_mask
        self.thresh = thresh

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mask = None
        if self.background is not None:
            diff = cv2.absdiff(self.background, gray)
            _, fg = cv2.threshold(diff, self.thresh, 255, cv2.THRESH_BINARY)
            mask = fg
        if self.roi_mask is not None:
            mask = mask & self.roi_mask if mask is not None else self.roi_mask.copy()
        return mask if mask is not None else gray
