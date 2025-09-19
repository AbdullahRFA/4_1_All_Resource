import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.filters import sobel

# Step 1: Load RGB image
rgb_image = img_as_float(data.astronaut())  # Normalized to [0, 1]

# Step 2: Split into R, G, B channels
R = rgb_image[:, :, 0]
G = rgb_image[:, :, 1]
B = rgb_image[:, :, 2]

# Step 3: Apply edge detection to each channel
R_edge = sobel(R)
G_edge = sobel(G)
B_edge = sobel(B)

# Step 4: Recombine edge-detected channels into RGB
edge_rgb = np.stack((R_edge, G_edge, B_edge), axis=2)

# Step 5: Create purely grayscale edge map
edge_gray = (R_edge + G_edge + B_edge) / 3

# Step 6: Display results
fig, axes = plt.subplots(1, 6, figsize=(24, 5))

axes[0].imshow(rgb_image)
axes[0].set_title('Original RGB')
axes[0].axis('off')

axes[1].imshow(R_edge, cmap='gray')
axes[1].set_title('Red Channel Edges')
axes[1].axis('off')

axes[2].imshow(G_edge, cmap='gray')
axes[2].set_title('Green Channel Edges')
axes[2].axis('off')

axes[3].imshow(B_edge, cmap='gray')
axes[3].set_title('Blue Channel Edges')
axes[3].axis('off')

axes[4].imshow(edge_rgb)
axes[4].set_title('Combined Edge RGB')
axes[4].axis('off')

axes[5].imshow(edge_gray, cmap='gray')
axes[5].set_title('Grayscale Edge Map')
axes[5].axis('off')

plt.tight_layout()
plt.show()