from abc import ABC, abstractmethod
from typing import List, Tuple

class BoundingBox:
    """Структура для прямоугольника и контура"""
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int, contour=None):
        self.x_min = x_min
        self.y_min = y_min
        self.x_max = x_max
        self.y_max = y_max
        self.contour = contour  # для классического детектора, если нужен угол

class Detector(ABC):
    @abstractmethod
    def detect(self, frame) -> List[BoundingBox]:
        """
        Находит объекты на кадре и возвращает список BoundingBox
        """
        pass
