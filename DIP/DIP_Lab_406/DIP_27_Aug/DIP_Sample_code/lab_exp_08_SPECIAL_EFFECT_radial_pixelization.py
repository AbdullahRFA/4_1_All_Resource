import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

# Load image
image = img_as_float(data.astronaut())
height, width, _ = image.shape

# Define center (you can change this)
center_x, center_y = width // 2, height // 2

# Compute distance of each pixel from center
Y, X = np.indices((height, width))
distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

# Define radial band size
band_size = 10  # pixels per ring
max_radius = int(np.max(distances))
num_bands = max_radius // band_size + 1

# Create output image
radial_pixelized = np.zeros_like(image)

# Apply pixelization per radial band
for band in range(num_bands):
    r_min = band * band_size
    r_max = (band + 1) * band_size
    mask = (distances >= r_min) & (distances < r_max)
    
    for c in range(3):  # R, G, B channels
        avg_color = np.mean(image[:, :, c][mask])
        radial_pixelized[:, :, c][mask] = avg_color

# Show original and radial pixelized image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(radial_pixelized)
axes[1].set_title('Radial Pixelized Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()