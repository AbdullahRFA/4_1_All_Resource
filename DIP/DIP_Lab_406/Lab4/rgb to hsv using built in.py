import matplotlib.pyplot as plt
import numpy as np
from skimage import io, color, exposure

# ------------------ Load and Normalize RGB Image ------------------
img_rgb = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff")

if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0

# ------------------ Convert to HSV & Equalize V ------------------
img_hsv = color.rgb2hsv(img_rgb)
V_original = img_hsv[:, :, 2].copy()

# Equalize V channel
img_hsv[:, :, 2] = exposure.equalize_hist(img_hsv[:, :, 2])
V_equalized = img_hsv[:, :, 2]

# Convert back to RGB
img_rgb_eq = color.hsv2rgb(img_hsv)

# ------------------ Plot All Outputs in One Window ------------------
fig, axes = plt.subplots(4, 3, figsize=(20, 24))
fig.suptitle("Digital Image Processing - RGB Channels, Histograms, and HSV Equalization", fontsize=16)

# ------------------ Row 1: Original RGB and Histogram ------------------
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB Image")
axes[0, 0].axis('off')

colors = ('r', 'g', 'b')
for i, color_name in enumerate(colors):
    hist, _ = np.histogram(img_rgb[:, :, i].flatten(), bins=256, range=[0, 1])
    axes[0, 1].plot(hist, color=color_name)
axes[0, 1].set_title("RGB Histograms")
axes[0, 1].set_xlim(0, 255)

axes[0, 2].imshow(img_rgb_eq)
axes[0, 2].set_title("RGB After V Equalization")
axes[0, 2].axis('off')

# ------------------ Row 2: R, G, B Channel Images ------------------
channel_titles = ['Red Channel', 'Green Channel', 'Blue Channel']
for i in range(3):
    axes[1, i].imshow(img_rgb[:, :, i], cmap='gray')
    axes[1, i].set_title(channel_titles[i])
    axes[1, i].axis('off')

# ------------------ Row 3: R, G, B Histograms ------------------
for i in range(3):
    hist, _ = np.histogram(img_rgb[:, :, i].flatten(), bins=256, range=[0, 1])
    axes[2, i].plot(hist, color=colors[i])
    axes[2, i].set_title(f"{channel_titles[i]} Histogram")
    axes[2, i].set_xlim(0, 255)

# ------------------ Row 4: V Channels and Their Histograms ------------------
axes[3, 0].imshow(V_original, cmap='gray')
axes[3, 0].set_title("Original V Channel")
axes[3, 0].axis('off')

axes[3, 1].imshow(V_equalized, cmap='gray')
axes[3, 1].set_title("Equalized V Channel")
axes[3, 1].axis('off')

hist_orig_v, _ = np.histogram(V_original.flatten(), bins=256, range=[0, 1])
hist_eq_v, _ = np.histogram(V_equalized.flatten(), bins=256, range=[0, 1])
axes[3, 2].plot(hist_orig_v, label='Original V', color='gray')
axes[3, 2].plot(hist_eq_v, label='Equalized V', color='black')
axes[3, 2].set_title("V Channel Histograms")
axes[3, 2].legend()
axes[3, 2].set_xlim(0, 255)

plt.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
