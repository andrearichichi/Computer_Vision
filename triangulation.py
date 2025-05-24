import json
import numpy as np

# === Keypoint labels (COCO-style 18) ===
keypoint_labels = [
    "Hips", "RHip", "RKnee", "RAnkle", "RFoot",
    "LHip", "LKnee", "LAnkle", "LFoot",
    "Spine", "Neck", "Head",
    "RShoulder", "RElbow", "RHand",
    "LShoulder", "LElbow", "LHand"
]

# === Funzione di triangolazione SVD ===
def triangulate_points(points2d, projection_matrices):
    A = []
    for cam, (x, y) in points2d.items():
        P = projection_matrices[cam]
        A.append(x * P[2] - P[0])
        A.append(y * P[2] - P[1])
    A = np.array(A)
    _, _, Vt = np.linalg.svd(A)
    X = Vt[-1]
    return (X[:3] / X[3])

# === Funzione per triangolare un frame (prima annotazione da ciascuna cam) ===
def triangulate_single_frame():
    import os

    # === Carica le projection matrices da JSON
    with open("dataset/projection_matrices.json", "r") as f:
        raw_proj = json.load(f)
    projection_matrices = {
        cam: np.array(matrix, dtype=np.float64) for cam, matrix in raw_proj.items()
    }

    # === Mappa outX -> path json
    json_files = {
        "out2": "dataset/camera_json/out2.json",
        "out5": "dataset/camera_json/out5.json",
        "out8": "dataset/camera_json/out8.json",
        "out13": "dataset/camera_json/out13.json"
    }

    keypoints_by_cam = {}

    for cam, path in json_files.items():
        if not os.path.exists(path):
            print(f"\u274c JSON non trovato: {path}")
            continue

        with open(path) as f:
            data = json.load(f)

        anns = data.get("annotations", [])
        if anns:
            kpts = np.array(anns[0]["keypoints"]).reshape(-1, 3)[:, :2]
            keypoints_by_cam[cam] = kpts

    keypoints_3d = []
    visible_labels = []

    for i, label in enumerate(keypoint_labels):
        pts2d = {}
        for cam, kpts in keypoints_by_cam.items():
            if i < len(kpts) and kpts[i][0] > 0 and cam in projection_matrices:
                pts2d[cam] = kpts[i]
        if len(pts2d) >= 2:
            point_3d = triangulate_points(pts2d, projection_matrices)
            keypoints_3d.append(point_3d)
            visible_labels.append(label)

    return np.array(keypoints_3d) / 1000.0, visible_labels
