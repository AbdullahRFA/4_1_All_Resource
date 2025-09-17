import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Step 1: Load grayscale image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Replace with your own grayscale image.")
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Step 2: Add synthetic periodic noise (horizontal stripes)
rows, cols = image.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
freq_x, freq_y = 8, 0
noise = 0.2 * np.sin(2 * np.pi * (freq_x * X / cols + freq_y * Y / rows))
noisy_image = np.clip(image + noise, 0, 1)

# Step 3: FFT of noisy image
F = fft2(noisy_image)
F_shifted = fftshift(F)
magnitude_spectrum = np.log1p(np.abs(F_shifted))

# Step 4: Create notch filter mask
mask = np.ones_like(F_shifted)
center_row, center_col = rows // 2, cols // 2
notch_radius = 5

notch_coords = [
    (center_row, center_col + freq_x),
    (center_row, center_col - freq_x)
]

for r, c in notch_coords:
    rr, cc = np.ogrid[:rows, :cols]
    mask_area = (rr - r)**2 + (cc - c)**2 <= notch_radius**2
    mask[mask_area] = 0

# Step 5: Apply notch filter
F_filtered = F_shifted * mask
F_inv = ifft2(ifftshift(F_filtered))
restored_image = np.real(F_inv)

# Step 6: Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = ['Original Image', 'Noisy Image', 'Magnitude Spectrum', 'Restored Image']
images = [image, noisy_image, magnitude_spectrum, restored_image]

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
