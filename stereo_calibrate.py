import cv2
import numpy as np
import os, sys

# ---------- параметры шахматной доски ----------
BOARD_SIZE   = (9, 6)          # внутренних углов
SQUARE_SIZE  = 0.025           # 25 mm в метрах
MAX_FRAMES   = 15

# папка для снимков
out_dir = "calib_frames"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)   # склейка потоков
assert cap.isOpened()

objp = np.zeros((BOARD_SIZE[0]*BOARD_SIZE[1],3), np.float32)
objp[:,:2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1,2)
objp *= SQUARE_SIZE

obj_points = []   # 3D точки
img_points_L = [] # 2D левого
img_points_R = [] #    правого
frame_id = 0

print("Нажимайте <space>, когда углы найдены. <q> — выйти.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    h, w = frame.shape[:2]
    L = frame[:, :w//2]
    R = frame[:, w//2:]

    grayL = cv2.cvtColor(L, cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(R, cv2.COLOR_BGR2GRAY)

    okL, cornersL = cv2.findChessboardCorners(grayL, BOARD_SIZE,
                                              cv2.CALIB_CB_ADAPTIVE_THRESH +
                                              cv2.CALIB_CB_NORMALIZE_IMAGE)
    okR, cornersR = cv2.findChessboardCorners(grayR, BOARD_SIZE,
                                              cv2.CALIB_CB_ADAPTIVE_THRESH +
                                              cv2.CALIB_CB_NORMALIZE_IMAGE)

    vis = np.hstack([L.copy(), R.copy()])
    if okL:
        cv2.drawChessboardCorners(vis, BOARD_SIZE, cornersL, okL)
    if okR:
        cv2.drawChessboardCorners(vis, BOARD_SIZE, cornersR, okR)

    cv2.putText(vis, f"captured: {len(obj_points)}/{MAX_FRAMES}",
                (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("stereo capture", vis)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        sys.exit()
    if key == ord(' ') and okL and okR:
        obj_points.append(objp)
        img_points_L.append(cornersL)
        img_points_R.append(cornersR)
        cv2.imwrite(f"{out_dir}/L_{frame_id:02d}.png", L)
        cv2.imwrite(f"{out_dir}/R_{frame_id:02d}.png", R)
        frame_id += 1
        print(f"Saved pair #{frame_id}")
        if frame_id >= MAX_FRAMES:
            break

cap.release(); cv2.destroyAllWindows()

# ---------- калибровка двух камер ----------
retL, KL, DL, _, _ = cv2.calibrateCamera(
    obj_points, img_points_L, grayL.shape[::-1], None, None)
retR, KR, DR, _, _ = cv2.calibrateCamera(
    obj_points, img_points_R, grayR.shape[::-1], None, None)

flags = (cv2.CALIB_FIX_INTRINSIC)
criteria = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)

retval, KL, DL, KR, DR, R, T, E, F = cv2.stereoCalibrate(
    obj_points, img_points_L, img_points_R,
    KL, DL, KR, DR, grayL.shape[::-1],
    criteria=criteria, flags=flags)

print("R =\n", R)
print("T =\n", T.T)

# ---------- выпрямление -------------
R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
    KL, DL, KR, DR, grayL.shape[::-1], R, T, alpha=0)

mapL1, mapL2 = cv2.initUndistortRectifyMap(
    KL, DL, R1, P1, grayL.shape[::-1], cv2.CV_16SC2)
mapR1, mapR2 = cv2.initUndistortRectifyMap(
    KR, DR, R2, P2, grayR.shape[::-1], cv2.CV_16SC2)

np.savez("stereo_maps.npz",
         KL=KL, KR=KR, DL=DL, DR=DR, R=R, T=T,
         mapL1=mapL1, mapL2=mapL2,
         mapR1=mapR1, mapR2=mapR2, Q=Q)
print("Файлы карты сохранены ⇒ stereo_maps.npz")
