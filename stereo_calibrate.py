"""
Стереокалибровка для «side-by-side» USB-камеры.
Сохраняет stereo_calib.npz с картами выпрямления под КОНКРЕТНЫЙ режим камеры.

Управление:
  - SPACE  : попытаться распознать шахматку и сохранить кадр в набор
  - 'v'    : включить/выключить визуализацию найденных углов
  - 'q'    : выход и запуск калибровки (если кадров достаточно)

  python stereo_calibrate.py --cam 1 --pattern 9x6 --square 25
"""

import argparse
import time
import cv2
import numpy as np

# ---------- утилиты видеорежима ----------

def force_mode(cap, w=640, h=480, fps=30, use_dshow=False, cam_index=None):
    """Жёстко выставить режим. Возвращает (W, H) фактические."""
    if use_dshow:
        cap.release()
        assert cam_index is not None
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

# ---------- парсер аргументов ----------

ap = argparse.ArgumentParser(description="Stereo calibration for side-by-side camera")
ap.add_argument("--cam", type=int, default=0, help="индекс камеры")
ap.add_argument("--backend", type=str, default="MSMF", choices=["MSMF", "DSHOW"])
ap.add_argument("--width",  type=int, default=640, help="общая ширина кадра (левая+правая)")
ap.add_argument("--height", type=int, default=480, help="высота кадра")
ap.add_argument("--fps",    type=int, default=30)
ap.add_argument("--pattern", type=str, default="9x6", help="внутренние углы (cols x rows), напр. 9x6")
ap.add_argument("--square",  type=float, default=25.0, help="размер клетки, мм")
ap.add_argument("--out",     type=str, default="stereo_calib.npz")
args = ap.parse_args()

# распарсим размер шахматки
cols, rows = map(int, args.pattern.lower().split("x"))
pattern_size = (cols, rows)
square = float(args.square)

# инициализация камеры
backend = cv2.CAP_MSMF if args.backend.upper() == "MSMF" else cv2.CAP_DSHOW
cap = cv2.VideoCapture(args.cam, backend)
if not cap.isOpened():
    raise RuntimeError("Не удалось открыть камеру")

W, H = force_mode(cap, args.width, args.height, args.fps, use_dshow=False, cam_index=args.cam)
if (W, H) != (args.width, args.height):
    print("[warn] MSMF не дал требуемый режим, пробуем DSHOW…")
    W, H = force_mode(cap, args.width, args.height, args.fps, use_dshow=True, cam_index=args.cam)

print(f"[info] Работаем в режиме {W}x{H} (в половинке {W//2}x{H})")

# подготовим шаблон 3D-точек (0,0,0)..(cols-1,rows-1,0) в мм
objp = np.zeros((rows*cols, 3), np.float32)
objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

# накопители точек
objpoints = []   # 3D точки (одни и те же для обеих камер)
imgpointsL = []  # 2D точки левой половины
imgpointsR = []  # 2D точки правой половины

viz = True
saved = 0
print("[hint] Нажимайте SPACE для добавления кадра, 'v' — включить/выключить отрисовку углов, 'q' — завершить.")

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    half_w = W // 2
    left  = frame[:, :half_w]
    right = frame[:, half_w:]

    grayL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    foundL, cornersL = cv2.findChessboardCorners(grayL, pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)
    foundR, cornersR = cv2.findChessboardCorners(grayR, pattern_size,
                                                 flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE)

    if foundL:
        cornersL = cv2.cornerSubPix(grayL, cornersL, (11,11), (-1,-1), criteria)
    if foundR:
        cornersR = cv2.cornerSubPix(grayR, cornersR, (11,11), (-1,-1), criteria)

    showL = left.copy()
    showR = right.copy()
    if viz and foundL:
        cv2.drawChessboardCorners(showL, pattern_size, cornersL, foundL)
    if viz and foundR:
        cv2.drawChessboardCorners(showR, pattern_size, cornersR, foundR)

    cv2.putText(showL, f"found: {foundL}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if foundL else (0,0,255), 2)
    cv2.putText(showR, f"found: {foundR}", (8,24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if foundR else (0,0,255), 2)

    cv2.imshow("Left",  showL)
    cv2.imshow("Right", showR)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('v'):
        viz = not viz
    elif key == 32:  # SPACE
        if foundL and foundR:
            objpoints.append(objp.copy())
            imgpointsL.append(cornersL)
            imgpointsR.append(cornersR)
            saved += 1
            print(f"[save] Кадр добавлен ({saved}).")
        else:
            print("[warn] Шахматка не найдена на обеих половинках — кадр не сохранён.")
    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

if saved < 8:
    raise RuntimeError(f"Недостаточно кадров для калибровки: {saved} (нужно ≥ 8)")

# одиночная калибровка каждой половины
img_size_half = (W//2, H)
retL, KL, DL, rvecsL, tvecsL = cv2.calibrateCamera(objpoints, imgpointsL, img_size_half, None, None)
retR, KR, DR, rvecsR, tvecsR = cv2.calibrateCamera(objpoints, imgpointsR, img_size_half, None, None)

# стереокалибровка (фиксируем внутренние параметры)
flags = (cv2.CALIB_FIX_INTRINSIC)
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
retval, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpointsL, imgpointsR, KL, DL, KR, DR, img_size_half,
    criteria=criteria_stereo, flags=flags
)

print(f"RMS stereo error: {retval:.6f}")

# выпрямление
R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(KL, DL, KR, DR, img_size_half, R, T, flags=cv2.CALIB_ZERO_DISPARITY)
map1x, map1y = cv2.initUndistortRectifyMap(KL, DL, R1, P1, img_size_half, cv2.CV_32FC1)
map2x, map2y = cv2.initUndistortRectifyMap(KR, DR, R2, P2, img_size_half, cv2.CV_32FC1)

np.savez_compressed(
    args.out,
    image_size=np.array([W, H], np.int32),
    half_size=np.array([img_size_half[0], img_size_half[1]], np.int32),
    KL=KL, DL=DL, KR=KR, DR=DR,
    R=R, T=T, E=E, F=F,
    R1=R1, R2=R2, P1=P1, P2=P2, Q=Q,
    map1x=map1x, map1y=map1y, map2x=map2x, map2y=map2y,
    swap_halves=False,       # можно изменить вручную после, если половинки перепутаны
    flip_right=False         # поставить True, если правая половина зеркальная
)

print(f"[done] Готово. Сохранено в {args.out}")
