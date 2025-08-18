#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Стереокалибровка для «side-by-side» USB-камеры.
Сохраняет stereo_calib.npz с картами выпрямления
Управление:
  SPACE — сохранить кадр (если условия выполнены / в fast-режиме просто found on both)
  A     — авто-съёмка вкл/выкл
  V     — включить/выключить отрисовку углов
  Z     — удалить последний сохранённый
  Q     — калибровка и выход

  python stereo_calibrate.py --cam 1 --backend MSMF --fourcc AUTO --width 2560 --height 720 --fps 30 --pattern 12x13 --square 25 --downscale 0.5 --fast
"""

import argparse
import time
import cv2
import numpy as np

# -------------------- открытие камеры (MSMF/DSHOW, MJPG/YUY2) --------------------

def _open_with_backend(cam_index, backend_tag):
    backend = cv2.CAP_MSMF if backend_tag == "MSMF" else cv2.CAP_DSHOW
    cap = cv2.VideoCapture(cam_index, backend)
    return cap, backend_tag

def force_mode(cap, w=2560, h=720, fps=30, fourcc="AUTO", cam_index=None, backend_tag="MSMF"):
    assert cam_index is not None

    def set_mode(tag):
        if tag is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*tag))
        cap.set(cv2.CAP_PROP_FPS, fps)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    tried = []
    if fourcc in ("AUTO", "MJPG"):
        set_mode("MJPG"); tried.append("MJPG")
    if fourcc == "YUY2":
        set_mode("YUY2"); tried.append("YUY2")

    t0 = time.time()
    while time.time() - t0 < 3.0:
        ok, fr = cap.read()
        if ok and fr is not None:
            H, W = fr.shape[:2]
            fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
            tag = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])
            print(f"[force_mode] actual frame: {W}x{H}, fourcc={tag}, backend={backend_tag}, tried={tried}")
            return W, H
        time.sleep(0.05)

    if fourcc == "AUTO":
        alt = "YUY2" if "MJPG" in tried else "MJPG"
        print(f"[warn] no frames on {tried}, fallback FOURCC → {alt}")
        set_mode(alt); tried.append(alt)
        t1 = time.time()
        while time.time() - t1 < 2.0:
            ok, fr = cap.read()
            if ok and fr is not None:
                H, W = fr.shape[:2]
                fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC))
                tag = "".join([chr((fourcc_int >> 8*i) & 0xFF) for i in range(4)])
                print(f"[force_mode] fallback OK: {W}x{H}, fourcc={tag}, backend={backend_tag}")
                return W, H
            time.sleep(0.05)

    raise RuntimeError("Камера не отдаёт кадры в заданном режиме")

def open_camera(cam_index, backend_prefer="MSMF", width=2560, height=720, fps=30, fourcc="AUTO"):
    cap, used = _open_with_backend(cam_index, backend_prefer)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру (backend={backend_prefer})")
    try:
        W, H = force_mode(cap, width, height, fps, fourcc, cam_index, used)
        return cap, (W, H), used
    except Exception as e:
        print(f"[warn] {e} → пробуем другой backend")
        cap.release()

    alt = "DSHOW" if backend_prefer == "MSMF" else "MSMF"
    cap, used = _open_with_backend(cam_index, alt)
    if not cap.isOpened():
        raise RuntimeError(f"Не удалось открыть камеру (backend={alt})")
    W, H = force_mode(cap, width, height, fps, fourcc, cam_index, used)
    return cap, (W, H), used

# -------------------- утилиты детекции и качества --------------------

def detect_corners_fast(gray, pattern_size, downscale=0.5, use_sb_fallback=True, draw=False, dst_img=None):
    """
    Быстрый поиск углов:
      1) уменьшаем серый кадр (downscale) для скорости,
      2) пробуем классический findChessboardCorners(+subpix),
      3) если не нашёл и включён fallback — пробуем findChessboardCornersSB,
      4) найденные углы масштабируем обратно в размер исходной половинки.
    """
    cols, rows = pattern_size
    h, w = gray.shape[:2]

    # даунскейл ради скорости
    if downscale < 1.0:
        resized = cv2.resize(gray, (int(w*downscale), int(h*downscale)), interpolation=cv2.INTER_AREA)
        scale_x = w / float(resized.shape[1])
        scale_y = h / float(resized.shape[0])
    else:
        resized = gray
        scale_x = scale_y = 1.0

    # 1) быстрый классический
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ok, corners = cv2.findChessboardCorners(resized, (cols, rows), flags=flags)

    # 2) если не нашёл — SB (медленнее, но устойчивее)
    if not ok and use_sb_fallback:
        try:
            sb_flags = (cv2.CALIB_CB_NORMALIZE_IMAGE |
                        cv2.CALIB_CB_EXHAUSTIVE |
                        cv2.CALIB_CB_ACCURACY)
            ok, corners = cv2.findChessboardCornersSB(resized, (cols, rows), flags=sb_flags)
        except Exception:
            ok = False
            corners = None

    if not ok:
        return False, None

    # sub-pixel уточнение — в даунскейле
    if corners is not None and corners.dtype != np.float32:
        corners = corners.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-3)
    corners = cv2.cornerSubPix(resized, corners, (11,11), (-1,-1), criteria)

    # масштабируем назад к исходному размеру половинки
    corners[:,0,0] *= scale_x
    corners[:,0,1] *= scale_y

    # опционально рисуем (в исходном масштабе, если передали dst_img)
    if draw and dst_img is not None:
        cv2.drawChessboardCorners(dst_img, (cols, rows), corners, True)

    return True, corners

def laplacian_var(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def chessboard_coverage(corners, img_shape):
    h, w = img_shape[:2]
    hull = cv2.convexHull(corners.reshape(-1,1,2))
    area = cv2.contourArea(hull)
    return float(area) / float(w*h + 1e-9)

def min_border_margin(corners, img_shape):
    h, w = img_shape[:2]
    xs = corners[:,0,0]; ys = corners[:,0,1]
    margin = min(xs.min(), ys.min(), (w-1)-xs.max(), (h-1)-ys.max())
    return margin / max(w, h)

def pose_signature(corners, img_shape):
    h, w = img_shape[:2]
    c = corners.reshape(-1,2)
    cx, cy = c[:,0].mean()/w, c[:,1].mean()/h
    scale = np.linalg.norm(c.max(0)-c.min(0)) / np.hypot(w,h)
    return np.array([cx, cy, scale], dtype=np.float32)

def is_diverse(sig, sigs, thr=0.08):
    if len(sigs) == 0:
        return True
    d = np.linalg.norm(sigs - sig, axis=1).min()
    return d > thr

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser(description="Stereo calibration for side-by-side camera (fast)")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--backend", type=str, default="MSMF", choices=["MSMF", "DSHOW"])
    ap.add_argument("--fourcc", type=str, default="AUTO", choices=["AUTO", "MJPG", "YUY2"])
    ap.add_argument("--width",  type=int, default=2560)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps",    type=int, default=30)
    ap.add_argument("--pattern", type=str, default="12x13", help="Внутренние углы: colsxrows, напр. 12x13")
    ap.add_argument("--square",  type=float, default=25.0, help="Размер клетки, мм")
    ap.add_argument("--out",     type=str, default="stereo_calib.npz")
    ap.add_argument("--downscale", type=float, default=0.5, help="Масштаб для детектора (0.5 = в 2 раза меньше)")
    ap.add_argument("--fast", action="store_true", help="Без проверок качества — сохранять, если найдено на обеих половинках")
    # quality thresholds (используются, если --fast не задан)
    ap.add_argument("--sharp_min", type=float, default=60.0)
    ap.add_argument("--cov_min",   type=float, default=0.08)
    ap.add_argument("--cov_max",   type=float, default=0.70)
    ap.add_argument("--margin_min",type=float, default=0.01)
    args = ap.parse_args()

    cols, rows = map(int, args.pattern.lower().split("x"))
    pattern_size = (cols, rows)
    square = float(args.square)

    cap, (W, H), used_backend = open_camera(
        cam_index=args.cam,
        backend_prefer=args.backend,
        width=args.width, height=args.height, fps=args.fps,
        fourcc=args.fourcc
    )
    print(f"[info] Работаем в режиме {W}x{H} (половинка {W//2}x{H}), backend={used_backend}")
    half_w, half_h = W//2, H

    # 3D сетка (в плоскости Z=0)
    objp = np.zeros((rows*cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square

    objpoints, imgpointsL, imgpointsR = [], [], []
    posesigsL, posesigsR = [], []

    viz, auto, saved = True, False, 0
    print("[hint] SPACE=save  A=auto  V=viz  Z=undo  Q=calibrate")

    cv2.namedWindow("Left"); cv2.namedWindow("Right")
    t_prev = time.time()
    not_found_streak = 0
    reported_hint = False

    def can_use_pair(grayL, grayR, cL, cR):
        if args.fast:
            return True, "OK(fast)"
        sharp = 0.5*(laplacian_var(grayL)+laplacian_var(grayR))
        cov = 0.5*(chessboard_coverage(cL, grayL.shape)+chessboard_coverage(cR, grayR.shape))
        marg = min(min_border_margin(cL, grayL.shape), min_border_margin(cR, grayR.shape))
        sigL = pose_signature(cL, grayL.shape); sigR = pose_signature(cR, grayR.shape)
        diverse = is_diverse(sigL, np.array(posesigsL)) and is_diverse(sigR, np.array(posesigsR))
        if sharp < args.sharp_min: return False, f"sharp={sharp:.0f} < {args.sharp_min}"
        if not (args.cov_min <= cov <= args.cov_max): return False, f"cov={cov:.2f}∉[{args.cov_min},{args.cov_max}]"
        if marg < args.margin_min: return False, f"margin={marg:.3f} < {args.margin_min}"
        if not diverse: return False, "pose not diverse"
        return True, "OK"

    def save_pair(cL, cR, sigL, sigR):
        nonlocal saved
        objpoints.append(objp.copy())
        imgpointsL.append(cL); imgpointsR.append(cR)
        posesigsL.append(sigL); posesigsR.append(sigR)
        saved += 1
        print(f"[save] кадр добавлен ({saved}).")

    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            print("[warn] пустой кадр"); continue
        if frame.shape[1] != W or frame.shape[0] != H:
            print(f"[error] режим изменился на {frame.shape[1]}x{frame.shape[0]} (ожидалось {W}x{H}). Выход.")
            break

        left, right = frame[:, :half_w], frame[:, half_w:]
        grayL, grayR = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

        foundL, cL = detect_corners_fast(grayL, pattern_size, downscale=args.downscale, use_sb_fallback=True)
        foundR, cR = detect_corners_fast(grayR, pattern_size, downscale=args.downscale, use_sb_fallback=True)

        showL, showR = left.copy(), right.copy()
        msg = ""
        if foundL and foundR:
            not_found_streak = 0
            if viz:
                cv2.drawChessboardCorners(showL, pattern_size, cL, True)
                cv2.drawChessboardCorners(showR, pattern_size, cR, True)
            ok_pair, msg = can_use_pair(grayL, grayR, cL, cR)
            if auto and ok_pair:
                sigL = pose_signature(cL, grayL.shape); sigR = pose_signature(cR, grayR.shape)
                save_pair(cL, cR, sigL, sigR)
        else:
            not_found_streak += 1
            if not reported_hint and not_found_streak in (60, 120):
                print("[hint] Шахматка не находится долго. Проверьте, что указаны ВНУТРЕННИЕ углы.")
                print("       Например, при 12×13 клетках нужно --pattern 11x12, а не 12x13.")
                reported_hint = True

        hud = "SPACE:save  A:auto  V:viz  Z:undo  Q:calibrate"
        fps = 1.0 / max(1e-6, (time.time() - t_prev)); t_prev = time.time()
        cv2.putText(showL, hud, (8, showL.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(showR, hud, (8, showR.shape[0]-12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255,255,255), 2)
        cv2.putText(showL, f"{fps:.1f} FPS", (8, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
        cv2.putText(showL, f"foundL={foundL}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if foundL else (0,0,255), 2)
        cv2.putText(showR, f"foundR={foundR}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if foundR else (0,0,255), 2)
        if msg:
            cv2.putText(showL, msg, (8, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0) if "OK" in msg else (0,0,255), 2)

        cv2.imshow("Left", showL); cv2.imshow("Right", showR)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('v'):
            viz = not viz
        elif key == ord('a'):
            auto = not auto
            print(f"[auto] {'ON' if auto else 'OFF'}")
        elif key == ord('z') and saved > 0:
            objpoints.pop(); imgpointsL.pop(); imgpointsR.pop()
            posesigsL.pop(); posesigsR.pop(); saved -= 1
            print(f"[undo] удалён последний, осталось {saved}")
        elif key == 32:  # SPACE
            if foundL and foundR:
                ok_pair, msg2 = can_use_pair(grayL, grayR, cL, cR)
                if ok_pair:
                    sigL = pose_signature(cL, grayL.shape); sigR = pose_signature(cR, grayR.shape)
                    save_pair(cL, cR, sigL, sigR)
                else:
                    print(f"[warn] кадр не сохранён: {msg2}")
            else:
                print("[warn] углы не найдены на обеих половинках")
        elif key == ord('q'):
            break

    cap.release(); cv2.destroyAllWindows()

    if saved < 8:
        raise RuntimeError(f"Недостаточно кадров для калибровки: {saved} (нужно ≥ 8)")

    # -------------------- калибровка и сохранение --------------------

    img_size_half = (half_w, half_h)
    calib_flags = cv2.CALIB_RATIONAL_MODEL

    print("[info] Калибровка левой…")
    retL, KL, DL, rL, tL = cv2.calibrateCamera(objpoints, imgpointsL, img_size_half, None, None, flags=calib_flags)
    print("[info] Калибровка правой…")
    retR, KR, DR, rR, tR = cv2.calibrateCamera(objpoints, imgpointsR, img_size_half, None, None, flags=calib_flags)

    criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 200, 1e-6)
    flags_st = cv2.CALIB_FIX_INTRINSIC | cv2.CALIB_RATIONAL_MODEL
    print("[info] Стерео-калибровка…")
    retval, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
        objpoints, imgpointsL, imgpointsR, KL, DL, KR, DR, img_size_half,
        criteria=criteria, flags=flags_st
    )
    print(f"[info] RMS stereo error: {retval:.6f}")

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(KL, DL, KR, DR, img_size_half, R, T,
                                                      flags=cv2.CALIB_ZERO_DISPARITY)
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
        swap_halves=False,
        flip_right=False
    )
    print(f"[done] Готово. Сохранено в {args.out}")

if __name__ == "__main__":
    main()
