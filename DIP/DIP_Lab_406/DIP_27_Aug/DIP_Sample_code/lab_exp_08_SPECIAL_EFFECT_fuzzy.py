import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float

# Load image
image = img_as_float(data.astronaut())
height, width, _ = image.shape

# Define neighborhood size
window_size = 7
pad = window_size // 2

# Pad image to handle borders
padded_image = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')

# Create output image
fuzzy_image = np.zeros_like(image)

# Apply random neighborhood replacement
for y in range(height):
    for x in range(width):
        # Random offset within window
        dy = np.random.randint(-pad, pad + 1)
        dx = np.random.randint(-pad, pad + 1)
        fuzzy_image[y, x] = padded_image[y + pad + dy, x + pad + dx]

# Show original and fuzzy image
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(fuzzy_image)
axes[1].set_title('Fuzzy Image (Random Neighborhood)')
axes[1].axis('off')

plt.tight_layout()
plt.show()