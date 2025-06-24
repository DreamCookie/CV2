import cv2
import argparse
from pathlib import Path
import time
import yaml
from detector.contour_detector import ContourDetector
from detector.nn_detector import NNDetector

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

def process_frame(frame, detector):
    boxes = detector.detect(frame)
    for b in boxes:
        cv2.rectangle(frame,
                      (b.x_min, b.y_min),
                      (b.x_max, b.y_max),
                      (0, 255, 0), 2)
        print(f"Box: ({b.x_min}, {b.y_min}) → ({b.x_max}, {b.y_max})")
    return frame

def main():
    parser = argparse.ArgumentParser(
        description="демка - камера/папка с изображениями"
    )
    parser.add_argument(
        "--config", "-c",
        type=Path,
        default=Path(__file__).parent / "config" / "detection.yaml",
        help="path to detection.yaml"
    )
    parser.add_argument(
        "--input-dir", "-i",
        type=Path,
        default=None,
        help="if set, process all .jpg/.png in this folder"
    )
    parser.add_argument(
        "--camera", "-d",
        type=int,
        default=0,
        help="camera ID (if --input-dir not set)"
    )
    parser.add_argument(
        "--use-nn",
        action="store_true",
        help="use NNDetector instead of ContourDetector"
    )
    args = parser.parse_args()

    # ЗАГРУЖАЕМ КОНФИГ В CFG
    cfg = load_config(args.config)

    # СОЗДАЁМ ДЕТЕКТОР ИЗ CFG
    detector = make_detector(cfg, use_nn=args.use_nn)

    if args.input_dir:
        # Обработка папки с картинками
        img_paths = sorted(args.input_dir.glob("*.[jp][pn]g"))
        for img_path in img_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue
            out = process_frame(frame, detector)
            cv2.imshow("Detection", out)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
        cv2.destroyAllWindows()

    else:
        # Захват с камеры
        cap = cv2.VideoCapture(args.camera)
        if not cap.isOpened():
            print(f"Cannot open camera #{args.camera}")
            return
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            start = time.time()
            out = process_frame(frame, detector)
            fps = 1.0 / (time.time() - start)
            cv2.putText(out, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.imshow("Detection", out)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
