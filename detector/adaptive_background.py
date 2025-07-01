import cv2
import numpy as np

class AdaptiveBackground:
    """
    Адаптивный алгоритм вычитания фона на основе MOG2
    Позволяет автоматически обновлять модель фона и получать маску движущихся/новых объектов
    """
    def __init__(self,
                 history: int = 500,
                 var_threshold: float = 16,
                 detect_shadows: bool = True):
        # history — длина истории кадров для обучения модели
        # var_threshold — порог дисперсии для определения фона
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=detect_shadows
        )

    def apply(self, frame: np.ndarray) -> np.ndarray:
        """
        Применить фоновой вычитатель к кадру
        param frame - BGR-изображение
        return - бинарная маска переднего плана (0 — фон, >0 — объекты/тени)
        """
        # получаем 1-канальную маску с тенями (127)
        mask = self.subtractor.apply(frame)
        # избавляемся от теней и переводим все >200 в 255, иначе 0
        _, fg = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        return fg