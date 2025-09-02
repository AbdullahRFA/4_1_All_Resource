import matplotlib.pyplot as plt
from skimage import io, exposure
import numpy as np

# Load RGB image
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'
rgb_image = io.imread(image_path)

# Ensure image is in float format
rgb_image = rgb_image.astype(float)

# Set threshold value
threshold_value = 50

# Normalize if needed
if np.max(rgb_image) > 255:
    rgb_image = rgb_image / np.max(rgb_image)
    threshold_value = threshold_value / 255.0

# Apply threshold to each channel
binary_mask_r = rgb_image[:, :, 0] < threshold_value
binary_mask_g = rgb_image[:, :, 1] < threshold_value
binary_mask_b = rgb_image[:, :, 2] < threshold_value

# Combine masks (logical OR: highlight if any channel is below threshold)
combined_mask = binary_mask_r | binary_mask_g | binary_mask_b

# Rescale for display
thresholded_image = exposure.rescale_intensity(combined_mask.astype(float), out_range=(0.0, 1.0))

# Plot original and thresholded images
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(rgb_image / np.max(rgb_image))  # Normalize for display
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(thresholded_image, cmap='gray')
axes[1].set_title('Thresholded RGB Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()