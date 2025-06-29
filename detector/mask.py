import cv2
import numpy as np

class BackgroundMasker:

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
            if mask is None:
                mask = self.roi_mask.copy()
            else:
                mask = cv2.bitwise_and(mask, self.roi_mask)
        return mask if mask is not None else gray