import yaml
from pathlib import Path

def load_yaml(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

cfg = load_yaml(Path(__file__).parent.parent / 'config' / 'detection.yaml')
contour_cfg = cfg['contour']
nms_iou = cfg['nms']['iou_threshold']
nn_cfg = cfg['nn']
