import matplotlib.pyplot as plt
from skimage import io, exposure, color
import numpy as np

# Load image
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.05.tiff'
image = io.imread(image_path)

# Convert to grayscale if needed
gray_image = color.rgb2gray(image) if image.ndim == 3 else image

# Determine dynamic range
min_val, max_val = np.min(gray_image), np.max(gray_image)

# Set threshold based on image range
if max_val <= 255:
    threshold_value = 50.0
    binary_mask = gray_image < threshold_value
else:
    # Normalize image to [0, 1] and adjust threshold accordingly
    gray_image = gray_image / max_val
    threshold_value = 50 / 255.0
    binary_mask = gray_image < threshold_value

# Rescale binary mask for display
thresholded_image = exposure.rescale_intensity(binary_mask.astype(float), out_range=(0.0, 1.0))

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Single Thresholded Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()



