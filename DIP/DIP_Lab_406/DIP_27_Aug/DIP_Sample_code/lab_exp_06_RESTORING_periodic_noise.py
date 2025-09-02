import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Load and normalize image
image = img_as_float(data.camera())

# Add synthetic periodic noise (sinusoidal pattern)
rows, cols = image.shape
x = np.arange(cols)
y = np.arange(rows)
X, Y = np.meshgrid(x, y)
freq_x, freq_y = 8, 0  # Horizontal stripes
noise = 0.2 * np.sin(2 * np.pi * (freq_x * X / cols + freq_y * Y / rows))
noisy_image = image + noise
noisy_image = np.clip(noisy_image, 0, 1)

# FFT of noisy image
F = fft2(noisy_image)
F_shifted = fftshift(F)
magnitude_spectrum = np.log(1 + np.abs(F_shifted))

# Create notch filter mask
mask = np.ones_like(F_shifted)
center_row, center_col = rows // 2, cols // 2
notch_radius = 5

# Coordinates of noise peaks (manually chosen based on freq_x)
notch_coords = [
    (center_row, center_col + freq_x),
    (center_row, center_col - freq_x)
]

for r, c in notch_coords:
    rr, cc = np.ogrid[:rows, :cols]
    mask_area = (rr - r)**2 + (cc - c)**2 <= notch_radius**2
    mask[mask_area] = 0

# Apply notch filter
F_filtered = F_shifted * mask
F_inv = ifft2(ifftshift(F_filtered))
restored_image = np.real(F_inv)

# Plot results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
titles = ['Original Image', 'Noisy Image', 'Magnitude Spectrum', 'Restored Image']
images = [image, noisy_image, magnitude_spectrum, restored_image]

for ax, img, title in zip(axes.flat, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()