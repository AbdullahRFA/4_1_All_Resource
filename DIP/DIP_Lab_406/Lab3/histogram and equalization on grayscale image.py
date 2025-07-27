import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from skimage import exposure, img_as_ubyte

# Load the image
img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")

# Convert RGB image to grayscale using luminance method
gray = 0.2989 * img1[:, :, 0] + 0.5870 * img1[:, :, 1] + 0.1140 * img1[:, :, 2]
gray = img_as_ubyte(gray / gray.max())  # Normalize to uint8 (0–255)

# Histogram of grayscale image
hist, bins = np.histogram(gray.flatten(), bins=256, range=[0, 256])

# Histogram Equalization
equalized = exposure.equalize_hist(gray)
equalized_uint8 = img_as_ubyte(equalized)

# Histogram of equalized image
equalized_hist, bins_eq = np.histogram(equalized_uint8.flatten(), bins=256, range=[0, 256])

# Plotting
plt.figure(figsize=(10, 8))

# Original Grayscale Image
plt.subplot(2, 2, 1)
plt.imshow(gray, cmap='gray')
plt.title('Original Grayscale')
plt.axis('off')

# Histogram of Original Grayscale
plt.subplot(2, 2, 2)
plt.plot(hist, color='black')
plt.title('Histogram (Original)')

# Equalized Image
plt.subplot(2, 2, 3)
plt.imshow(equalized_uint8, cmap='gray')
plt.title('Equalized Grayscale')
plt.axis('off')

# Histogram of Equalized
plt.subplot(2, 2, 4)
plt.plot(equalized_hist, color='black')
plt.title('Histogram (Equalized)')

plt.tight_layout()
plt.show()