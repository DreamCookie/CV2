import cv2
import argparse
import time
import math
from pathlib import Path
import yaml
import rclpy
from detector.adaptive_background import AdaptiveBackground
from detector.contour_detector import ContourDetector
from detector.nn_detector import NNDetector
from detector.preprocessing import Preprocessor
#from detector.coord_transform import CoordTransformer
from detector.ros_coord_transform import RosCoordTransformer

# глобальные переменные для дедупликации
DEDUP_THRESHOLD_MM = 10.0  # порог в мм для новизны объекта
prev_centers = []          # список предыдущих координат (X,Y)


def load_config(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def make_detector(cfg, use_nn: bool):
    """Создать экземпляр детектора"""
    if use_nn and 'nn' in cfg:
        nn_cfg = cfg['nn']
        return NNDetector(model_path=nn_cfg['model_path'],
                          conf_thresh=nn_cfg.get('conf_thresh', 0.5))
    contour_cfg = cfg['contour']
    return ContourDetector(
        min_area=contour_cfg['min_area'],
        blur_ksize=tuple(contour_cfg['blur_ksize']),
        threshold_params=contour_cfg['threshold_params'],
        nms_iou_thresh=cfg['nms']['iou_threshold']
    )


def init_pipeline(config_path: Path, use_ros: bool=False):
    """
    Инициализация адаптивного фонового вычитания, препроцессора и координатного трансформера.
    """
    # адаптивное вычитание фона
    masker = AdaptiveBackground(history=300, var_threshold=25)
    # препроцессинг CLAHE + blur
    prep = Preprocessor()
    # координатный трансформер
    coord = None
    if use_ros:
        rclpy.init()
        coord = RosCoordTransformer(config_path)
    return masker, prep, coord


def process_frame(frame, detector, masker, prep, coord, cfg, mode, use_ros: bool=False):
    """
    Обработать кадр - фон, препроцессинг, ROI-маски, детекция, дедупликация, 
    перевод в координаты, отрисовка
    """
    global prev_centers
    # адаптивный foreground
    fg = masker.apply(frame)
    # ROI для текущего режима
    roi_masks = cfg.get('roi_masks', {}).get(mode, [])
    for x1, y1, x2, y2 in roi_masks:
        fg[y1:y2, x1:x2] = 0
    # препроцессинг
    _ = prep.apply(frame)
    # детекция на полном фрейме
    boxes = detector.detect(frame)

    new_centers = []
    # проходим по найденным боксам
    for b in boxes:
        # центр в пикселях
        cx = (b.x_min + b.x_max) * 0.5
        cy = (b.y_min + b.y_max) * 0.5
        # угол ориентации, если есть контур
        angle = None
        if hasattr(b, 'contour') and b.contour is not None:
            rect = cv2.minAreaRect(b.contour)
            angle = rect[2]
        # перевод координат
        if use_ros and coord:
            pt = coord.pixels_to_robot(cx, cy)
            if pt is None:
                continue
            X, Y, Z = pt
        else:
            X, Y = cx, cy
            Z = 0.0
        # дедупликация
        is_new = True
        for px, py in prev_centers:
            if math.hypot(X - px, Y - py) < DEDUP_THRESHOLD_MM:
                is_new = False
                break
        if not is_new:
            continue
        new_centers.append((X, Y))
        # вывод и отрисовка
        label = f"{X:.0f},{Y:.0f}"
        if angle is not None:
            label += f", θ={angle:.0f}°"
        print("New object:", label)
        cv2.rectangle(frame, (b.x_min, b.y_min), (b.x_max, b.y_max), (0,255,0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0,0,255), -1)
        cv2.putText(frame, label, (b.x_min, b.y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    prev_centers = new_centers
    return frame


def main():
    parser = argparse.ArgumentParser(description="CV-Detector: camera or folder")
    parser.add_argument("-c", "--config", type=Path,
                        default=Path(__file__).parent/"config"/"detection.yaml",
                        help="path to detection.yaml")
    parser.add_argument("-i", "--input-dir", type=Path, default=None,
                        help="folder with images to process")
    parser.add_argument("-d", "--camera", type=int, default=0,
                        help="camera ID (if not using folder)")
    parser.add_argument("--use-nn", action="store_true",
                        help="use neural detector YOLOv8")
    parser.add_argument("--use-ros", action="store_true",
                        help="use ROS2 TF for hand-eye calibration")
    parser.add_argument("--mode", type=str, choices=["workpieces","tools"],
                        default="workpieces",
                        help="which ROI mode to apply")
    args = parser.parse_args()

    # загрузка конфига и детектора
    cfg = load_config(args.config)
    detector = make_detector(cfg, use_nn=args.use_nn)
    masker, prep, coord = init_pipeline(args.config, use_ros=args.use_ros)

    # выбор roi-масок из конфига
    mode = args.mode

    # обработка изображений из папки
    if args.input_dir:
        img_paths = sorted(args.input_dir.glob("*.*g"))
        if not img_paths:
            print(f"No images in {args.input_dir}")
            return
        for path in img_paths:
            frame = cv2.imread(str(path))
            if frame is None:
                continue
            out = process_frame(frame, detector, masker, prep, coord, cfg, mode, args.use_ros)
            cv2.imshow("Detection", out)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()
        return

    # обработка с камеры
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Cannot open camera #{args.camera}")
        return
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out = process_frame(frame, detector, masker, prep, coord, cfg, mode, args.use_ros)
        cv2.imshow("Detection", out)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

'''
python src/main.py --input-dir test_img/

python src/main.py --input-dir test_img/ --use-nn

Обработка папки с ROS2-TF hand–eye калибровкой
python src/main.py --input-dir test_img/ --use-ros

обработка папки, нейросеть + ROS2
python src/main.py --input-dir test_img/ --use-nn --use-ros

Переключение режимов ROI
python src/main.py --input-dir test_img/ --mode tools
python src/main.py --input-dir test_img/ --mode workpieces

Пример для камеры №0, классика
python main.py --camera 0

Камера + нейросеть
python main.py --camera 0 --use-nn

Камера + ROS2 hand–eye
python src/main.py --camera 0 --use-ros

Камера + нейросеть + ROS2
python src/main.py --camera 0 --use-nn --use-ros
'''