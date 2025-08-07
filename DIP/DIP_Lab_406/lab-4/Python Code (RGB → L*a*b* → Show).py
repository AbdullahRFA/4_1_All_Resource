import matplotlib.pyplot as plt
from skimage import io, color
import numpy as np

# --------- Load RGB Image ---------
img_rgb = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0  # Normalize if in 0–255 range

# --------- Convert RGB to L*a*b ---------
img_lab = color.rgb2lab(img_rgb)

# --------- Separate L, a, b Channels ---------
L = img_lab[:, :, 0]
a = img_lab[:, :, 1]
b = img_lab[:, :, 2]

# --------- Merge L, a, b Channels Back ---------
merged_lab = np.stack((L, a, b), axis=2)

# --------- Convert Back to RGB ---------
converted_rgb = color.lab2rgb(merged_lab)

# --------- Display All Images ---------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Original RGB Image
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis('off')

# L* Channel
axes[0, 1].imshow(L, cmap='gray')
axes[0, 1].set_title("L* Channel")
axes[0, 1].axis('off')

# a* Channel
axes[0, 2].imshow(a, cmap='RdYlGn')
axes[0, 2].set_title("a* Channel")
axes[0, 2].axis('off')

# b* Channel
axes[1, 0].imshow(b, cmap='PuOr')
axes[1, 0].set_title("b* Channel")
axes[1, 0].axis('off')

# Merged L*a*b* Image (False colored)
axes[1, 1].imshow(merged_lab / np.max(merged_lab))  # for visualization only
axes[1, 1].set_title("Merged L*a*b* (Scaled View)")
axes[1, 1].axis('off')

# Reconverted RGB Image
axes[1, 2].imshow(converted_rgb)
axes[1, 2].set_title("Converted RGB from L*a*b*")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()
