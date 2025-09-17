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
height, width, _ = image_rgb.shape

# Step 2: Define neighborhood size
window_size = 7
pad = window_size // 2

# Step 3: Pad image to handle borders
padded_image = cv2.copyMakeBorder(image_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)

# Step 4: Create output image
fuzzy_image = np.zeros_like(image_rgb)

# Step 5: Apply random neighborhood replacement
for y in range(height):
    for x in range(width):
        dy = np.random.randint(-pad, pad + 1)
        dx = np.random.randint(-pad, pad + 1)
        fuzzy_image[y, x] = padded_image[y + pad + dy, x + pad + dx]

# Step 6: Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(fuzzy_image)
axes[1].set_title('Fuzzy Image (Random Neighborhood)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
