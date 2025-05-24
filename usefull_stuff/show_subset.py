import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from math import ceil
from glob import glob

def show_image_subset(image_dir, prefix="out2_", max_images=48):
    # Cerca sia PNG che JPG
    images = sorted(glob(os.path.join(image_dir, f"{prefix}*.png")))
    if len(images) == 0:
        images = sorted(glob(os.path.join(image_dir, f"{prefix}*.jpg")))

    if len(images) == 0:
        print(f"Nessuna immagine trovata con prefisso {prefix} in {image_dir}")
        return

    images = images[:max_images]
    cols = 8
    rows = ceil(len(images) / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(10, 5))


    for i, ax in enumerate(axs.flat):
        if i < len(images):
            img = mpimg.imread(images[i])
            ax.imshow(img)
            ax.set_title(os.path.basename(images[i]), fontsize=6)
            ax.axis('off')
        else:
            ax.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # 🔧 Modifica questi parametri se vuoi vedere un altro subset
    image_directory = "Mocap 4 v1 May 20 2025/train"
    camera_prefix = "out2_"  # Cambia in "out5_", "out8_", ecc.
    
    show_image_subset(image_directory, prefix=camera_prefix)
