import cv2
import numpy as np

cap = cv2.VideoCapture(1, cv2.CAP_MSMF)  # или CAP_DSHOW
assert cap.isOpened()

sgbm = cv2.StereoSGBM_create(
    minDisparity=0,
    numDisparities=16*8,
    blockSize=7,
    P1=8*7*7,
    P2=32*7*7,
    mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    uniquenessRatio=8,
    speckleWindowSize=100,
    speckleRange=64,
    disp12MaxDiff=1
)

FLIP_RIGHT = False  # попробуй True, если правая половинка зеркалит

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    w2 = w // 2
    left  = frame[:, :w2]
    right = frame[:, w2:]
    if FLIP_RIGHT:
        right = cv2.flip(right, 1)

    grayL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    # Немного выровнять контраст
    grayL = cv2.equalizeHist(grayL)
    grayR = cv2.equalizeHist(grayR)

    disp = sgbm.compute(grayL, grayR).astype(np.float32) / 16.0
    disp[disp <= 0] = np.nan

    dv = np.nan_to_num(disp, nan=0.0)
    dv = cv2.normalize(dv, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    dv = cv2.applyColorMap(dv, cv2.COLORMAP_JET)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    cv2.imshow("Disparity", dv)

    if (cv2.waitKey(1) & 0xFF) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
