import cv2
import argparse
import time
from pathlib import Path
import yaml
from detector.contour_detector import ContourDetector
from detector.nn_detector import NNDetector
from detector.calibration import PlaneCalibrator
from detector.mask import BackgroundMasker
from detector.preprocessing import Preprocessor
from detector.coord_transform import CoordTransformer

def load_config(path: Path):
    with path.open('r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def make_detector(cfg, use_nn: bool):
    if use_nn and 'nn' in cfg:
        nn_cfg = cfg['nn']
        return NNDetector(
            model_path=nn_cfg['model_path'],
            conf_thresh=nn_cfg.get('conf_thresh', 0.5)
        )
    else:
        contour_cfg = cfg['contour']
        nms_iou = cfg['nms']['iou_threshold']
        return ContourDetector(
            min_area=contour_cfg['min_area'],
            blur_ksize=tuple(contour_cfg['blur_ksize']),
            threshold_params=contour_cfg['threshold_params'],
            nms_iou_thresh=nms_iou
        )

def init_pipeline(first_frame, config_path):
    '''
    Калибровка плоскости по ArUco если на кадре есть метки

    Еще можно конечно замерить с помощью линейки параллельно осям камеры и сделать замер, тогда
    p1=(x1,y1), p2=(x2,y2)
    pixel_dist = np.hypot(x2 - x1, y2 - y1)
    real_dist_mm = xxx.x  # длина линейки
    scale_mm_per_px = real_dist_mm / pixel_dist

    '''
    calibrator = PlaneCalibrator(marker_length=100.0)  # мм
    if calibrator.calibrate(first_frame):
        print("Гомография рассчитана по ArUco-маркерам")
    else:
        print("ArUco-маркеры не найдены")

    # Фоновое вычитание по первому кадру
    gray0 = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    masker = BackgroundMasker(background=gray0)

    # Предобработка
    prep = Preprocessor()

    # Преобразование пикселей в мм
    coord = CoordTransformer(config_path=config_path, calibrator=calibrator)

    return calibrator, masker, prep, coord

def process_frame(frame, detector, masker, prep, coord):
    #  опционально- маска и препроцесс 
    fg   = masker.apply(frame)   # np.ndarray, 1 канал
    proc = prep.apply(frame)

    # детекция на полном BGR-кадре
    boxes = detector.detect(frame)

    # перевод координат и отрисовка
    for b in boxes:
        cx = (b.x_min + b.x_max) / 2.0
        cy = (b.y_min + b.y_max) / 2.0
        X, Y = coord.pixels_to_world(cx, cy)
        print(f"→ пиксели ({cx:.1f},{cy:.1f}) → мм ({X:.1f},{Y:.1f})")
        cv2.rectangle(frame, (b.x_min, b.y_min), (b.x_max, b.y_max), (0,255,0), 2)
        cv2.circle(frame, (int(cx), int(cy)), 3, (0,0,255), -1)
        cv2.putText(frame, f"{X:.0f},{Y:.0f}", (b.x_min, b.y_min-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    return frame

def main():
    parser = argparse.ArgumentParser(description="CV-Detector")
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=Path(__file__).parent / "config" / "detection.yaml",
        help="path to detection.yaml"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=None,
        help="process all images in a folder and exit"
    )
    parser.add_argument(
        "-d", "--camera",
        type=int,
        default=0,
        help="camera ID (if --input-dir not set)"
    )
    parser.add_argument(
        "--use-nn",
        action="store_true",
        help="use NNDetector (YOLOv8) instead of ContourDetector"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    detector = make_detector(cfg, use_nn=args.use_nn)

    # --- режим обработки папки с изображениями ---
    if args.input_dir:
        img_paths = sorted(args.input_dir.glob("*.*g"))
        if not img_paths:
            print(f"нет изображений в {args.input_dir}")
            return

        # инициализация по первому кадру
        first = cv2.imread(str(img_paths[0]))
        calibrator, masker, prep, coord = init_pipeline(first, args.config)

        for path in img_paths:
            frame = cv2.imread(str(path))
            out = process_frame(frame, detector, masker, prep, coord)
            cv2.imshow("Detection", out)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

        cv2.destroyAllWindows()
        return

    # --- режим работы с камерой ---
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print(f"Не удалось открыть камеру #{args.camera}")
        return

    # захват первого кадра для инициализации
    ret, first = cap.read()
    if not ret:
        print("не удалось прочитать первый кадр с камеры")
        return
    calibrator, masker, prep, coord = init_pipeline(first, args.config)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("упал кадр")
            break

        t0 = time.time()
        out = process_frame(frame, detector, masker, prep, coord)
        fps = 1.0 / (time.time() - t0)
        cv2.putText(out, f"FPS: {fps:.1f}", (10,30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,0), 2)

        cv2.imshow("Detection", out)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()