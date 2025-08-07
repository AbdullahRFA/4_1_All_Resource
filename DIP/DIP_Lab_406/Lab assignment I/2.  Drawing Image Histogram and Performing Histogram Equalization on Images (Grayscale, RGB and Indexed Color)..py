import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure, img_as_ubyte
from sklearn.cluster import KMeans

# ---------- Load Grayscale and RGB images ----------
gray_img = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/5.3.01.tiff", as_gray=True)
rgb_img = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")

# Normalize RGB image
if rgb_img.dtype == np.uint8:
    rgb_img = rgb_img / 255.0

# ---------- Simulate Indexed Color Image using KMeans ----------
pixels = rgb_img.reshape(-1, 3)
kmeans = KMeans(n_clusters=8, random_state=0).fit(pixels)
indexed_pixels = kmeans.labels_.reshape(rgb_img.shape[:2])
indexed_img = color.label2rgb(indexed_pixels, image=rgb_img, bg_label=0)

# ---------- Histogram Equalization ----------
gray_eq = exposure.equalize_hist(gray_img)
rgb_eq = np.zeros_like(rgb_img)
for i in range(3):
    rgb_eq[:, :, i] = exposure.equalize_hist(rgb_img[:, :, i])

indexed_eq = exposure.equalize_hist(img_as_ubyte(color.rgb2gray(indexed_img)))

# ---------- Helper Function to Plot Histogram ----------
def plot_histogram(image, ax, title, color='black'):
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])
    ax.plot(hist, color=color)
    ax.set_title(title)

# ---------- Plot All ----------
fig, axes = plt.subplots(3, 4, figsize=(18, 12))

# Grayscale
axes[0, 0].imshow(gray_img, cmap='gray')
axes[0, 0].set_title("Original Grayscale")
axes[0, 0].axis('off')

axes[0, 1].imshow(gray_eq, cmap='gray')
axes[0, 1].set_title("Equalized Grayscale")
axes[0, 1].axis('off')

plot_histogram(gray_img, axes[0, 2], "Grayscale Histogram")
plot_histogram(gray_eq, axes[0, 3], "Equalized Grayscale Histogram")

# RGB
axes[1, 0].imshow(rgb_img)
axes[1, 0].set_title("Original RGB")
axes[1, 0].axis('off')

axes[1, 1].imshow(rgb_eq)
axes[1, 1].set_title("Equalized RGB")
axes[1, 1].axis('off')

for i, c in enumerate(['r', 'g', 'b']):
    plot_histogram(rgb_img[:, :, i], axes[1, 2], "RGB Histogram", color=c)
    plot_histogram(rgb_eq[:, :, i], axes[1, 3], "Equalized RGB Histogram", color=c)

# Indexed
axes[2, 0].imshow(indexed_img)
axes[2, 0].set_title("Simulated Indexed Image (KMeans)")
axes[2, 0].axis('off')

axes[2, 1].imshow(indexed_eq, cmap='gray')
axes[2, 1].set_title("Equalized Indexed Image")
axes[2, 1].axis('off')

plot_histogram(img_as_ubyte(color.rgb2gray(indexed_img)), axes[2, 2], "Indexed Histogram")
plot_histogram(indexed_eq, axes[2, 3], "Equalized Indexed Histogram")

plt.tight_layout()
plt.show()