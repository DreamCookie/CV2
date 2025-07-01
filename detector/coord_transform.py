import yaml
from pathlib import Path
from typing import Tuple

class CoordTransformer:
    """
    Линейное преобразование пикселей → мм
    с коэффициентом scale_mm_per_px из конфига
    """
    def __init__(self, config_path: Path, calibrator=None):
        with config_path.open('r', encoding='utf-8') as f:
            cfg = yaml.safe_load(f)
        self.scale_mm_per_px = cfg['contour'].get('scale_mm_per_px', 1.0)

    def pixels_to_world(self, cx: float, cy: float) -> Tuple[float, float]:
        X_mm = cx * self.scale_mm_per_px
        Y_mm = cy * self.scale_mm_per_px
        return X_mm, Y_mm
