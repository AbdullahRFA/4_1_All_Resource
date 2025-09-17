import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'
rgb_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)

# Convert BGR to RGB if needed
if rgb_image.ndim == 3:
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

# Convert to float for processing
rgb_image = rgb_image.astype(np.float32)

# Set threshold value
threshold_value = 50.0

# Normalize if needed
max_val = np.max(rgb_image)
if max_val > 255:
    rgb_image /= max_val
    threshold_value /= 255.0

# Apply threshold to each channel
binary_mask_r = rgb_image[:, :, 0] < threshold_value
binary_mask_g = rgb_image[:, :, 1] < threshold_value
binary_mask_b = rgb_image[:, :, 2] < threshold_value

# Combine masks (logical OR)
combined_mask = binary_mask_r | binary_mask_g | binary_mask_b

# Normalize mask for display
thresholded_image = cv2.normalize(combined_mask.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX)

# Normalize original image for display
normalized_rgb = rgb_image / np.max(rgb_image)

# Display using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(normalized_rgb)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Thresholded RGB Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
