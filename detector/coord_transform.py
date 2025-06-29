import yaml
import numpy as np
from pathlib import Path
from typing import Tuple

class CoordTransformer:
    def __init__(self, config_path: Path, calibrator=None):
        cfg = yaml.safe_load(open(config_path, 'r', encoding='utf-8'))
        self.scale = cfg['contour'].get('scale_mm_per_px', 1.0)
        self.calibrator = calibrator

    def pixels_to_world(self, cx: float, cy: float) -> Tuple[float,float]:
        if self.calibrator and self.calibrator.H is not None:
            X, Y = self.calibrator.image_to_plane(cx, cy)
        else:
            X, Y = cx * self.scale, cy * self.scale
        return X, Y