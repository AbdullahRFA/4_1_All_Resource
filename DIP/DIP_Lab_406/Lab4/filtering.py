import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy.ndimage import convolve, gaussian_filter

# ------------------ Load and Normalize RGB Image ------------------
img = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff")
if img.dtype == np.uint8:
    img = img / 255.0

# ------------------ Convert RGB to Grayscale ------------------
gray = 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]
# gray = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/5.2.08.tiff")

# ------------------ Define Filters ------------------
# Average filter
avg_kernel = np.ones((3, 3)) / 9

# High-pass filter
high_pass_kernel = np.array([[-1, -1, -1],
                             [-1,  8, -1],
                             [-1, -1, -1]])

# ------------------ Filtering Grayscale Image ------------------
gray_avg = convolve(gray, avg_kernel)
gray_high = np.clip(convolve(gray, high_pass_kernel), 0, 1)
gray_gauss = gaussian_filter(gray, sigma=1)

# ------------------ Filtering RGB Image Channel-wise ------------------
def filter_rgb(img, kernel, method="convolve", sigma=1):
    filtered = np.zeros_like(img)
    for i in range(3):  # R, G, B
        if method == "convolve":
            filtered[:, :, i] = convolve(img[:, :, i], kernel)
        elif method == "gaussian":
            filtered[:, :, i] = gaussian_filter(img[:, :, i], sigma=sigma)
    return np.clip(filtered, 0, 1)

rgb_avg = filter_rgb(img, avg_kernel, method="convolve")
rgb_high = filter_rgb(img, high_pass_kernel, method="convolve")
rgb_gauss = filter_rgb(img, None, method="gaussian", sigma=1)

# ------------------ Display All Images ------------------
plt.figure(figsize=(16, 12))

# --- Original ---
plt.subplot(3, 3, 1)
plt.imshow(img)
plt.title("Original RGB")
plt.axis('off')

plt.subplot(3, 3, 4)
plt.imshow(gray, cmap='gray')
plt.title("Original Grayscale")
plt.axis('off')

# --- Average ---
plt.subplot(3, 3, 2)
plt.imshow(rgb_avg)
plt.title("RGB - Average Filter")
plt.axis('off')

plt.subplot(3, 3, 5)
plt.imshow(gray_avg, cmap='gray')
plt.title("Gray - Average Filter")
plt.axis('off')

# --- High-pass ---
plt.subplot(3, 3, 3)
plt.imshow(rgb_high)
plt.title("RGB - High-Pass Filter")
plt.axis('off')

plt.subplot(3, 3, 6)
plt.imshow(gray_high, cmap='gray')
plt.title("Gray - High-Pass Filter")
plt.axis('off')

# --- Gaussian ---
plt.subplot(3, 3, 7)
plt.imshow(rgb_gauss)
plt.title("RGB - Gaussian Filter")
plt.axis('off')

plt.subplot(3, 3, 8)
plt.imshow(gray_gauss, cmap='gray')
plt.title("Gray - Gaussian Filter")
plt.axis('off')

plt.tight_layout()
plt.show()
