import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.color import rgb2hsv, rgb2lab

# Load RGB image
rgb_image = img_as_float(data.astronaut())

# Convert to HSV and LAB
hsv_image = rgb2hsv(rgb_image)
lab_image = rgb2lab(rgb_image)

# Extract channels
R, G, B = rgb_image[:, :, 0], rgb_image[:, :, 1], rgb_image[:, :, 2]
H, S, V = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
L, a, b = lab_image[:, :, 0], lab_image[:, :, 1], lab_image[:, :, 2]

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