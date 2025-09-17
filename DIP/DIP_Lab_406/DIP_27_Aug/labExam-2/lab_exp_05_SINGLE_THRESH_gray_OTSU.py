import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.05.tiff'
image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Convert to grayscale if needed
if image.ndim == 3:
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    gray_image = image.copy()

# Determine dynamic range
min_val, max_val = np.min(gray_image), np.max(gray_image)

# Normalize only if dynamic range exceeds [0, 255]
if max_val > 255:
    gray_image = gray_image.astype(np.float32) / max_val
    gray_image_scaled = (gray_image * 255).astype(np.uint8)  # Scale to [0, 255] for OpenCV thresholding
else:
    gray_image_scaled = gray_image.astype(np.uint8)

# Apply Otsu's threshold using OpenCV
otsu_thresh, binary_mask = cv2.threshold(gray_image_scaled, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Normalize binary mask for display
thresholded_image = cv2.normalize(binary_mask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

# Display using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title(f'Otsu Thresholded Image\nThreshold = {otsu_thresh:.4f}')
axes[1].axis('off')

plt.tight_layout()
plt.show()
