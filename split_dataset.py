import json

# Carica il file COCO
with open("Mocap v1i Coco Export/train/_annotations.coco.json", "r") as f:
    coco_data = json.load(f)

# ID delle telecamere da estrarre
camera_ids = [2, 5, 8, 13]

# Crea un dizionario per ogni telecamera
camera_annotations = {cam_id: {"images": [], "annotations": [], "categories": coco_data["categories"]} for cam_id in camera_ids}

# Filtra le immagini e le annotazioni per ogni telecamera
for image in coco_data["images"]:
    if image["id"] in camera_ids:
        # Aggiungi l'immagine al dizionario della telecamera corrispondente
        camera_annotations[image["id"]]["images"].append(image)

        # Aggiungi le annotazioni corrispondenti
        for annotation in coco_data["annotations"]:
            if annotation["image_id"] == image["id"]:
                camera_annotations[image["id"]]["annotations"].append(annotation)

# Scrivi i file JSON per ogni telecamera
for cam_id, data in camera_annotations.items():
    output_filename = f"camera_{cam_id}_annotations.json"
    with open(output_filename, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Creato il file: {output_filename}")
    
    # loll