import numpy as np
import matplotlib.pyplot as plt
from skimage import data, morphology
from skimage.filters import threshold_otsu

# Load grayscale image
image = data.coins()

# Apply thresholding to convert grayscale to binary
thresh = threshold_otsu(image)
binary_image = image > thresh

# Define structuring element
selem = morphology.disk(3)

# Morphological operations
eroded = morphology.erosion(binary_image, selem)
dilated = morphology.dilation(binary_image, selem)

# Plot the original and processed images
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].imshow(image, cmap='gray')
axes[0, 0].set_title('Original Grayscale Image')
axes[0, 0].axis('off')

axes[0, 1].imshow(binary_image, cmap='gray')
axes[0, 1].set_title('Binary Image (Thresholded)')
axes[0, 1].axis('off')

axes[1, 0].imshow(eroded, cmap='gray')
axes[1, 0].set_title('Eroded Image')
axes[1, 0].axis('off')

axes[1, 1].imshow(dilated, cmap='gray')
axes[1, 1].set_title('Dilated Image')
axes[1, 1].axis('off')

plt.tight_layout()
plt.show()
