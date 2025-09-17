import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert BGR to RGB and normalize to [0, 1]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Step 2: Single Scale Retinex function
def single_scale_retinex(img, sigma):
    retinex = np.zeros_like(img)
    for c in range(3):  # Apply to each channel
        blurred = cv2.GaussianBlur(img[:, :, c], (0, 0), sigma)
        retinex[:, :, c] = np.log1p(img[:, :, c]) - np.log1p(blurred)
    return retinex

# Step 3: Normalize and clip
def normalize_retinex(retinex):
    retinex = (retinex - np.min(retinex)) / (np.max(retinex) - np.min(retinex))
    return np.clip(retinex, 0, 1)

# Step 4: Apply SSR
sigma = 30
retinex_result = single_scale_retinex(image_rgb, sigma)
retinex_result = normalize_retinex(retinex_result)

# Step 5: Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image_rgb)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(retinex_result)
axes[1].set_title('Retinex Enhanced Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
