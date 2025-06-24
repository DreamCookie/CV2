from abc import ABC, abstractmethod
from typing import List, Tuple

class BoundingBox:
    def __init__(self, x_min: int, y_min: int, x_max: int, y_max: int, score: float = None):
        self.x_min, self.y_min = x_min, y_min
        self.x_max, self.y_max = x_max, y_max
        self.score = score

    def to_tuple(self):
        return (self.x_min, self.y_min, self.x_max, self.y_max)

class Detector(ABC):
    @abstractmethod
    def detect(self, frame) -> List[BoundingBox]:
        """
        Принимает BGR-кадр (numpy.ndarray)
        Возвращает список рамок в пикселях
        """
        pass
