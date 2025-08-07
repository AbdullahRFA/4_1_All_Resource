import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from skimage import io, img_as_ubyte

# ---------- Load RGB and Grayscale Images ----------
rgb_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff"
gray_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/5.3.01.tiff"

img_rgb = Image.open(rgb_path).convert('RGB')
img_gray = Image.open(gray_path).convert('L')  # Grayscale

# ---------- Save as PPM and PGM ----------
ppm_path = "converted_rgb.ppm"
pgm_path = "converted_gray.pgm"

img_rgb.save(ppm_path, format='PPM')
img_gray.save(pgm_path, format='PPM')  # Pillow uses 'PPM' for both PPM/PGM

# ---------- Convert RGB to Indexed Color Image ----------
rgb_np = np.asarray(img_rgb) / 255.0
pixels = rgb_np.reshape((-1, 3))

kmeans = KMeans(n_clusters=16, random_state=42)
labels = kmeans.fit_predict(pixels)
palette = (kmeans.cluster_centers_ * 255).astype(np.uint8)

indexed_img = palette[labels].reshape(rgb_np.shape)
indexed_img_uint8 = img_as_ubyte(indexed_img)

# ---------- Display All Images ----------
fig, axes = plt.subplots(2, 3, figsize=(18, 10))

# Original RGB
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB Image")
axes[0, 0].axis('off')

# Grayscale Image
axes[0, 1].imshow(img_gray, cmap='gray')
axes[0, 1].set_title("Original Grayscale Image")
axes[0, 1].axis('off')

# Indexed Image
axes[0, 2].imshow(indexed_img)
axes[0, 2].set_title("Indexed Color Image (KMeans)")
axes[0, 2].axis('off')

# PPM
ppm_img = Image.open(ppm_path)
axes[1, 0].imshow(ppm_img)
axes[1, 0].set_title("PPM Image")
axes[1, 0].axis('off')

# PGM
pgm_img = Image.open(pgm_path)
axes[1, 1].imshow(pgm_img, cmap='gray')
axes[1, 1].set_title("PGM Image")
axes[1, 1].axis('off')

# Indexed Image again
axes[1, 2].imshow(indexed_img_uint8)
axes[1, 2].set_title("Indexed Image (as uint8)")
axes[1, 2].axis('off')

plt.tight_layout()
plt.show()