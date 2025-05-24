import json
import os
from collections import defaultdict

# === Path ===
INPUT_JSON = "../Mocap 4 v1 May 20 2025/train/_annotations.coco.json"
OUTPUT_DIR = "dataset/camera_json"

# === Load COCO file ===
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

# === Init camera data containers ===
camera_data = defaultdict(lambda: {
    "info": data["info"],
    "licenses": data["licenses"],
    "categories": data["categories"],
    "images": [],
    "annotations": []
})

# === Map image_id to image entry and detect camera ===
image_id_to_camera = {}
for img in data["images"]:
    filename = img["file_name"]
    if "out2" in filename:
        cam = "out2"
    elif "out5" in filename:
        cam = "out5"
    elif "out8" in filename:
        cam = "out8"
    elif "out13" in filename:
        cam = "out13"
    else:
        continue  # skip unknown cams

    image_id_to_camera[img["id"]] = cam
    camera_data[cam]["images"].append(img)

# === Filter annotations by camera ===
for ann in data["annotations"]:
    image_id = ann["image_id"]
    cam = image_id_to_camera.get(image_id)
    if cam:
        camera_data[cam]["annotations"].append(ann)

# === Ensure output directory exists ===
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === Save each camera json ===
for cam, content in camera_data.items():
    output_path = os.path.join(OUTPUT_DIR, f"{cam}.json")
    with open(output_path, "w") as f:
        json.dump(content, f, indent=2)
    print(f"✅ Salvato: {output_path}")
