import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure, img_as_float

# Load sample RGB image
image = img_as_float(data.astronaut())  # RGB image

# Prepare an empty array to hold the transformed channels
fft_channels = []

# Apply DFT to each channel separately
for i in range(3):  # R, G, B channels
    channel = image[:, :, i]
    fft_image = np.fft.fft2(channel)
    fft_shifted = np.fft.fftshift(fft_image)
    fft_log_magnitude = np.log1p(np.abs(fft_shifted))
    fft_rescaled = exposure.rescale_intensity(fft_log_magnitude, out_range=(0.0, 1.0))
    fft_channels.append(fft_rescaled)

# Stack the channels back into an RGB image
fft_rgb = np.stack(fft_channels, axis=-1)

# Plot the original and transformed images side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(fft_rgb)
axes[1].set_title('DFT Magnitude Spectrum (RGB)')
axes[1].axis('off')

plt.tight_layout()
plt.show()