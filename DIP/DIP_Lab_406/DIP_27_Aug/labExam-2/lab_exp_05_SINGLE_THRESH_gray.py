import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.05.tiff'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)  # Preserves original bit depth

# Convert to grayscale if needed
if image.ndim == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image.copy()

# Determine dynamic range
min_val, max_val = np.min(gray_image), np.max(gray_image)

# Apply thresholding based on dynamic range
if max_val <= 255:
    threshold_value = 50.0
    binary_mask = gray_image < threshold_value
else:
    # Normalize to [0, 1] and adjust threshold
    gray_image = gray_image.astype(np.float32) / max_val
    threshold_value = 50 / 255.0
    binary_mask = gray_image < threshold_value

# Normalize binary mask for display
thresholded_image = cv2.normalize(binary_mask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

# Display using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Single Thresholded Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
