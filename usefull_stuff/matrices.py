import json
import numpy as np
import cv2
import os

# === Mappatura cam → file calibrazione ===
cam_calib_files = {
    "out2": "../dataset/camera_calib/calib_real/cam2_calib_real.json",
    "out5": "../dataset/camera_calib/calib_real/cam5_calib_real.json",
    "out8": "../dataset/camera_calib/calib_real/cam8_calib_real.json",
    "out13": "../dataset/camera_calib/calib_real/cam13_calib_real.json"
}

projection_matrices = {}

for cam, path in cam_calib_files.items():
    if not os.path.exists(path):
        print(f"❌ File non trovato: {path}")
        continue

    with open(path, "r") as f:
        calib = json.load(f)

    K = np.array(calib["mtx"], dtype=np.float64)
    tvec = np.array(calib["tvecs"], dtype=np.float64).reshape((3, 1))
    rvec = np.array(calib["rvecs"], dtype=np.float64).reshape((3, 1))

    # Converti rvec in matrice di rotazione (sarà identità se rvec == [0,0,0])
    R, _ = cv2.Rodrigues(rvec)

    # Costruisci matrice di proiezione P = K * [R | t]
    Rt = np.hstack((R, tvec))
    P = K @ Rt
    projection_matrices[cam] = P

# === Output ===
for cam, P in projection_matrices.items():
    print(f"\n📷 {cam}:\n{P}")
