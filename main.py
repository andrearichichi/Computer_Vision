import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from triangulation import triangulate_single_frame

# === Triangola il primo frame
points_3d, labels_3d = triangulate_single_frame()

# === Visualizza i keypoint 3D
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(points_3d[:, 0], points_3d[:, 1], points_3d[:, 2], c='red')

for i, label in enumerate(labels_3d):
    ax.text(points_3d[i, 0], points_3d[i, 1], points_3d[i, 2], label, fontsize=8)

ax.set_title("Triangolazione 3D - Primo frame")
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_zlabel("Z")
plt.tight_layout()
plt.show()
