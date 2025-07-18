import cv2
import numpy as np

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)      # или CAP_DSHOW
assert cap.isOpened()

# StereoSGBM для быстрой проверки глубины
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=16*6, blockSize=5,
    P1=8*3*5**2, P2=32*3*5**2, uniquenessRatio=10,
    speckleWindowSize=50, speckleRange=32
)

while True:
    ok, frame = cap.read()
    if not ok:
        break

    h, w = frame.shape[:2]
    left  = frame[:, :w//2]
    right = frame[:, w//2:]

    grayL = cv2.cvtColor(left,  cv2.COLOR_BGR2GRAY)
    grayR = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    disp = stereo.compute(grayL, grayR).astype(np.float32) / 16.0
    disp_vis = cv2.normalize(disp, None, 0, 255, cv2.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    cv2.imshow("Left", left)
    cv2.imshow("Right", right)
    cv2.imshow("Disparity", disp_vis)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
