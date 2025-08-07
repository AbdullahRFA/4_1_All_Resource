import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure

# 1. Load and Normalize RGB Image
img_rgb = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0  # normalize for skimage HSV functions

# 2. Convert RGB to HSV
img_hsv = color.rgb2hsv(img_rgb)

# 3. Separate H, S, V
H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

# 4. Equalize V channel
V_eq = exposure.equalize_hist(V)

# 5. Merge back to HSV with modified V
hsv_eq = np.stack((H, S, V_eq), axis=2)

# 6. Convert back to RGB
img_rgb_final = color.hsv2rgb(hsv_eq)

# 7. Plotting everything
fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Original RGB
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis('off')

# V Channel Before
axes[0, 1].imshow(V, cmap='gray')
axes[0, 1].set_title("Original V Channel")
axes[0, 1].axis('off')

# V Channel After Equalization
axes[0, 2].imshow(V_eq, cmap='gray')
axes[0, 2].set_title("Equalized V Channel")
axes[0, 2].axis('off')

# Final RGB
axes[1, 0].imshow(img_rgb_final)
axes[1, 0].set_title("Final RGB after V Equalization")
axes[1, 0].axis('off')

# Histogram: Original V
v_hist, _ = np.histogram(V.flatten(), bins=256, range=[0, 1])
axes[1, 1].plot(v_hist, color='black')
axes[1, 1].set_title("Histogram of V (Original)")

# Histogram: Equalized V
v_eq_hist, _ = np.histogram(V_eq.flatten(), bins=256, range=[0, 1])
axes[1, 2].plot(v_eq_hist, color='black')
axes[1, 2].set_title("Histogram of V (Equalized)")

plt.tight_layout()
plt.show()