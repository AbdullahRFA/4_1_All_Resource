import numpy as np
import matplotlib.pyplot as plt
from skimage import data, exposure

# Load sample image
image = data.camera()

# Compute the 2D Fourier Transform and shift the zero-frequency component to the center
fft_image = np.fft.fft2(image)
fft_shifted = np.fft.fftshift(fft_image)

# Apply logarithmic scaling for better visualization
fft_log_magnitude = np.log1p(np.abs(fft_shifted))

# Rescale intensity for display
fft_rescaled = exposure.rescale_intensity(fft_log_magnitude, out_range=(0.0, 1.0))

# Plot the original and transformed images side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Cameraman Image')
axes[0].axis('off')

axes[1].imshow(fft_rescaled, cmap='gray')
axes[1].set_title('DFT Magnitude Spectrum')
axes[1].axis('off')

plt.tight_layout()
plt.show()
