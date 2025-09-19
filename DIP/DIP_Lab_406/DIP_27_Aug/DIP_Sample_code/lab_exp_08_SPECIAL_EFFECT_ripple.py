import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.transform import warp

# Load image
image = img_as_float(data.astronaut())
height, width, _ = image.shape

# Ripple parameters
amplitude = 5       # pixels
frequency = 0.05    # ripple density
phase = 0           # phase shift
center_x, center_y = width // 2, height // 2

# Correct ripple function for skimage warp
def ripple(coords):
    # coords is shape (N, 2): [:, 0] = y, [:, 1] = x
    y = coords[:, 0]
    x = coords[:, 1]
    dx = x - center_x
    dy = y - center_y
    r = np.sqrt(dx**2 + dy**2)
    offset = amplitude * np.sin(frequency * r + phase)

    # Avoid division by zero
    r_safe = r + 1e-6
    x_new = x + (dx / r_safe) * offset
    y_new = y + (dy / r_safe) * offset

    # Return transformed coordinates as (N, 2)
    return np.column_stack((y_new, x_new))

# Apply ripple effect
rippled_image = warp(image, ripple, mode='reflect')

# Show result
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(rippled_image)
axes[1].set_title('Ripple Effect')
axes[1].axis('off')

plt.tight_layout()
plt.show()