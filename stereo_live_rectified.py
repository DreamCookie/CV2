"""
выпрямление и просмотр диспаратности на основе stereo_calib.npz

Клавиши:
  - 'q' — выход
"""

import argparse
import cv2
import numpy as np

def force_mode(cap, w=640, h=480, fps=30, use_dshow=False, cam_index=None):
    if use_dshow:
        cap.release()
        cap.open(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    ok, fr = cap.read()
    if not ok:
        raise RuntimeError("Камера не отдаёт кадры")
    H, W = fr.shape[:2]
    print(f"[force_mode] actual frame: {W}x{H}")
    return W, H

ap = argparse.ArgumentParser(description="Stereo live rectified viewer")
ap.add_argument("--cam", type=int, default=0)
ap.add_argument("--backend", type=str, default="MSMF", choices=["MSMF", "DSHOW"])
ap.add_argument("--fps", type=int, default=30)
ap.add_argument("--cfg", type=str, default="stereo_calib.npz")
args = ap.parse_args()

# загрузим калибровку
data = np.load(args.cfg)
Wc, Hc = data["image_size"]
half_w, half_h = data["half_size"]
map1x, map1y = data["map1x"], data["map1y"]
map2x, map2y = data["map2x"], data["map2y"]
swap_halves = bool(data["swap_halves"])
flip_right  = bool(data["flip_right"])

backend = cv2.CAP_MSMF if args.backend.upper() == "MSMF" else cv2.CAP_DSHOW
cap = cv2.VideoCapture(args.cam, backend)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

W, H = force_mode(cap, int(Wc), int(Hc), args.fps, use_dshow=False, cam_index=args.cam)
if (W, H) != (Wc, Hc):
    print("[warn] MSMF не дал режим из калибровки, пробуем DSHOW…")
    W, H = force_mode(cap, int(Wc), int(Hc), args.fps, use_dshow=True, cam_index=args.cam)

if (W, H) != (Wc, Hc):
    raise RuntimeError(f"Текущий режим {W}x{H} не совпадает с режимом калибровки {Wc}x{Hc}. "
                       "Переключите режим камеры или перекалибруйте в текущем режиме.")

print(f"[info] Половинка ожидается {half_w}x{half_h}")

# настроим SGBM
numDisp = 16 * 6
blockSize = 5
stereo = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=numDisp,
    blockSize=blockSize,
    P1=8*3*blockSize**2,
    P2=32*3*blockSize**2,
    uniquenessRatio=10,
    speckleWindowSize=50,
    speckleRange=32
)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    if frame.shape[1] != W or frame.shape[0] != H:
        # Защита от внезапной смены режима
        print("[warn] Камера сменила размер, выхожу.")
        break

    left  = frame[:, :W//2]
    right = frame[:, W//2:]

    if swap_halves:
        left, right = right, left

    if flip_right:
        right = cv2.flip(right, 1)

    if (left.shape[1] != half_w) or (left.shape[0] != half_h):
        # Несоответствие половинки сохранённым картам — показывать не будем
        black = np.zeros((half_h, half_w, 3), np.uint8)
        cv2.imshow("Rectified Left", black)
        cv2.imshow("Rectified Right", black)
        cv2.imshow("Disparity (color)", black)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        continue

    rectL = cv2.remap(left,  map1x, map1y, cv2.INTER_LINEAR)
    rectR = cv2.remap(right, map2x, map2y, cv2.INTER_LINEAR)

    grayL = cv2.cvtColor(rectL, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(rectR, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_masked = np.where(disp > 0, disp, 0)  # отсекаем отрицательные
    disp_vis = cv2.normalize(disp_masked, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)
    disp_color = cv2.applyColorMap(disp_vis, cv2.COLORMAP_JET)

    cv2.imshow("Rectified Left", rectL)
    cv2.imshow("Rectified Right", rectR)
    cv2.imshow("Disparity (color)", disp_color)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
