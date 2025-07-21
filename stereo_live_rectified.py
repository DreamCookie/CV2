import cv2, numpy as np

data = np.load("stereo_maps.npz")
mapL1, mapL2 = data["mapL1"], data["mapL2"]
mapR1, mapR2 = data["mapR1"], data["mapR2"]
Q = data["Q"]                   # 4×4 для глубины..

cap = cv2.VideoCapture(0, cv2.CAP_MSMF)
stereo = cv2.StereoSGBM_create(
    minDisparity=0, numDisparities=16*9, blockSize=7,
    P1=8*3*7**2, P2=32*3*7**2, uniquenessRatio=15,
    speckleWindowSize=100, speckleRange=32
)

while True:
    ret, frame = cap.read();          h, w = frame.shape[:2]
    L = frame[:, :w//2];  R = frame[:, w//2:]
    Lr = cv2.remap(L, mapL1, mapL2, cv2.INTER_LINEAR)
    Rr = cv2.remap(R, mapR1, mapR2, cv2.INTER_LINEAR)

    d = stereo.compute(cv2.cvtColor(Lr,cv2.COLOR_BGR2GRAY),
                       cv2.cvtColor(Rr,cv2.COLOR_BGR2GRAY)).astype(np.float32)/16
    depth = cv2.reprojectImageTo3D(d, Q)[:,:,2]    # Z в метрах, не перепутать..

    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = np.uint8(depth_vis)

    cv2.imshow("Depth (m, normalized)", depth_vis)
    if cv2.waitKey(1)&0xFF == ord('q'): break
