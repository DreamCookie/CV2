import cv2
import numpy as np
from typing import Tuple

class PlaneCalibrator:
    def __init__(self, marker_length: float, aruco_dict=cv2.aruco.DICT_6X6_1000):
        self.marker_length = marker_length  # мм
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict)
        self.parameters = cv2.aruco.DetectorParameters()
        self.H = None  # матрица гомографии

    def calibrate(self, frame: np.ndarray) -> bool:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is None or len(ids) < 4:
            return False
        obj_pts = []
        img_pts = []
        for corner, mid in zip(corners, ids.flatten()):
            c = corner[0].mean(axis=0)
            img_pts.append(c)
            if mid == 0:
                obj_pts.append([0, 0])
            elif mid == 1:
                obj_pts.append([self.marker_length, 0])
            elif mid == 2:
                obj_pts.append([self.marker_length, self.marker_length])
            elif mid == 3:
                obj_pts.append([0, self.marker_length])
        if len(obj_pts) < 4:
            return False
        obj_pts = np.array(obj_pts, dtype=np.float32)
        img_pts = np.array(img_pts, dtype=np.float32)
        self.H, _ = cv2.findHomography(img_pts, obj_pts)
        return True

    def image_to_plane(self, x: float, y: float) -> Tuple[float, float]:
        if self.H is None:
            raise RuntimeError("Гомография не вычислена")
        pt = np.array([[x, y, 1.0]], dtype=np.float32).T
        plane = self.H @ pt
        plane /= plane[2]
        return float(plane[0]), float(plane[1])