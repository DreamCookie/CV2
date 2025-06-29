import cv2
import numpy as np
from typing import Tuple

class Preprocessor:
    def __init__(self, use_clahe: bool = True, clahe_clip: float = 2.0, clahe_grid: Tuple[int,int] = (8,8)):
        self.use_clahe = use_clahe
        if self.use_clahe:
            self.clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=clahe_grid)

    def apply(self, frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5,5), 0)
        if self.use_clahe:
            blur = self.clahe.apply(blur)
        return blur