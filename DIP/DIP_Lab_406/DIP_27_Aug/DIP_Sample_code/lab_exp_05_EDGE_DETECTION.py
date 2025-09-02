import matplotlib.pyplot as plt
from skimage import io, color, filters
import numpy as np

# Load image
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.05.tiff'
image = io.imread(image_path)

# Convert to grayscale if needed
gray_image = color.rgb2gray(image) if image.ndim == 3 else image

# Apply edge detection filters
edges_prewitt = filters.prewitt(gray_image)
edges_sobel = filters.sobel(gray_image)
edges_roberts = filters.roberts(gray_image)

# Plot results
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(edges_prewitt, cmap='gray')
axes[1].set_title('Prewitt Edge')
axes[1].axis('off')

axes[2].imshow(edges_sobel, cmap='gray')
axes[2].set_title('Sobel Edge')
axes[2].axis('off')

axes[3].imshow(edges_roberts, cmap='gray')
axes[3].set_title('Roberts Edge')
axes[3].axis('off')

plt.tight_layout()
plt.show()