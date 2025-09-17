import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load grayscale image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Replace with your own grayscale image.")

# Step 2: Normalize to [0, 255] and convert to uint8
gray_image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# Step 3: Apply pseudo coloring using OpenCV's colormap
# Options include: COLORMAP_JET, COLORMAP_HOT, COLORMAP_VIRIDIS, etc.
pseudo_colored = cv2.applyColorMap(gray_image, cv2.COLORMAP_JET)
pseudo_colored_rgb = cv2.cvtColor(pseudo_colored, cv2.COLOR_BGR2RGB)

# Step 4: Display results using matplotlib
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(pseudo_colored_rgb)
axes[1].set_title('Pseudo Colored (Jet)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
