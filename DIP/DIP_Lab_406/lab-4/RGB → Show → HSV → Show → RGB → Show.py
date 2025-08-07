import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color

# ---------- 1. Load and Normalize RGB Image ----------
img_rgb = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")

# Normalize if image is in uint8
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0

# ---------- 2. Show Original RGB Image ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(img_rgb)
axes[0].set_title("Original RGB Image")
axes[0].axis('off')

# ---------- 3. Convert RGB → HSV ----------
img_hsv = color.rgb2hsv(img_rgb)

axes[1].imshow(img_hsv)
axes[1].set_title("Converted HSV Image (displayed in RGB)")
axes[1].axis('off')

# ---------- 4. Convert HSV → RGB ----------
img_rgb_converted = color.hsv2rgb(img_hsv)

axes[2].imshow(img_rgb_converted)
axes[2].set_title("Back to RGB from HSV")
axes[2].axis('off')

plt.tight_layout()
plt.show()