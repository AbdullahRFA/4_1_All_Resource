import matplotlib.pyplot as plt
from skimage import io, exposure, color
from skimage.filters import threshold_isodata
import numpy as np
import cv2

# Load image
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.05.tiff'
image = cv2.imread(image_path)
# print(image.ndim)

# Convert to grayscale if needed
gray_image = color.rgb2gray(image) if image.ndim == 3 else image

# Determine dynamic range
min_val, max_val = np.min(gray_image), np.max(gray_image)

# Normalize only if dynamic range exceeds [0, 255]
if max_val > 255:
    gray_image = gray_image / max_val  # Normalize to [0, 1]
    

# Apply ISODATA threshold
isodata_thresh = threshold_isodata(gray_image)
binary_mask = gray_image < isodata_thresh
# Rescale binary mask for display
thresholded_image = exposure.rescale_intensity(binary_mask.astype(float), out_range=(0.0, 1.0))

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title(f'ISODATA Thresholded Image\nThreshold = {isodata_thresh:.4f}')
axes[1].axis('off')

plt.tight_layout()
plt.show()