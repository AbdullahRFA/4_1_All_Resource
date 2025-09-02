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

# Apply thresholding based on dynamic range
if max_val <= 255:
    # Image in [0, 255] range
    lower_thresh, upper_thresh = 50, 100
    mask = (gray_image >= lower_thresh) & (gray_image <= upper_thresh)
else:
    # Normalize image to [0, 1] and adjust thresholds
    gray_image = gray_image / max_val
    lower_thresh, upper_thresh = 50 / 255.0, 100 / 255.0
    mask = (gray_image >= lower_thresh) & (gray_image <= upper_thresh)

# Rescale mask for display
thresholded_image = exposure.rescale_intensity(mask.astype(float), out_range=(0.0, 1.0))

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Double Thresholded Image [50, 100]')
axes[1].axis('off')

plt.tight_layout()
plt.show()