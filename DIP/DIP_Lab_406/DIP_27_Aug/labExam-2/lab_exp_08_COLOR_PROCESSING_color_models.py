import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert BGR to RGB and normalize to [0, 1]
rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Convert to HSV and LAB
hsv_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255.0
lab_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)

# Extract channels
R, G, B = cv2.split(rgb_image)
H, S, V = cv2.split(hsv_image)
L, a, b = cv2.split(lab_image)

# === First Window: Image Grid ===
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

image_grid = [
    [rgb_image, R, G, B],
    [rgb_image, H, S, V],
    [rgb_image, L, a, b]
]
titles_grid = [
    ['Original RGB', 'Red Channel', 'Green Channel', 'Blue Channel'],
    ['Original RGB', 'Hue Channel', 'Saturation Channel', 'Value Channel'],
    ['Original RGB', 'L* Channel', 'a* Channel', 'b* Channel']
]

for i in range(3):
    for j in range(4):
        img = image_grid[i][j]
        title = titles_grid[i][j]
        cmap = 'gray' if img.ndim == 2 else None
        axes[i, j].imshow(img, cmap=cmap)
        axes[i, j].set_title(title)
        axes[i, j].axis('off')

plt.tight_layout()
plt.show()

# === Second Window: Histogram Grid ===
fig, axes = plt.subplots(3, 3, figsize=(18, 12))

hist_channels = [
    [R, G, B],
    [H, S, V],
    [L, a, b]
]
hist_titles = [
    ['Red', 'Green', 'Blue'],
    ['Hue', 'Saturation', 'Value'],
    ['L*', 'a*', 'b*']
]
hist_colors = [
    ['red', 'green', 'blue'],
    ['purple', 'orange', 'gray'],
    ['black', 'teal', 'brown']
]

for i in range(3):
    for j in range(3):
        channel = hist_channels[i][j]
        title = hist_titles[i][j]
        color = hist_colors[i][j]
        ax = axes[i, j]
        ax.hist(channel.ravel(), bins=256, color=color, alpha=0.7)
        ax.set_title(f'{title} Histogram')
        ax.set_xlim([np.min(channel), np.max(channel)])

plt.tight_layout()
plt.show()
