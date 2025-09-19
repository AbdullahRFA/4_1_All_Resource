import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
import matplotlib

# Step 1: Load grayscale image
gray_image = img_as_float(data.camera())  # Normalized to [0, 1]

# Step 2: Use updated colormap access (Matplotlib 3.7+)
colormap = matplotlib.colormaps['jet']  # You can swap 'jet' with 'hot', 'viridis', etc.

# Step 3: Apply pseudo coloring
pseudo_colored = colormap(gray_image)[..., :3]  # Drop alpha channel

# Step 4: Display results
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(pseudo_colored)
axes[1].set_title('Pseudo Colored (Jet)')
axes[1].axis('off')

plt.tight_layout()
plt.show()