import cv2
import numpy as np

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
    lower_thresh, upper_thresh = 50, 100
    mask = cv2.inRange(gray_image, lower_thresh, upper_thresh)
else:
    # Normalize to [0, 1] float and adjust thresholds
    gray_image = gray_image.astype(np.float32) / max_val
    lower_thresh, upper_thresh = 50 / 255.0, 100 / 255.0
    mask = cv2.inRange(gray_image, lower_thresh, upper_thresh)

# Normalize mask for display
thresholded_image = cv2.normalize(mask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

# Display using matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Double Thresholded Image [50, 100]')
axes[1].axis('off')

plt.tight_layout()
plt.show()
