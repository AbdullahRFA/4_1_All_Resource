import cv2
import numpy as np

# Load RGB image (replace with your own if needed)
image = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.02.tiff")  # BGR format
if image is None:
    raise FileNotFoundError("Sample image not found. Replace with your own image path.")

# Convert to RGB and normalize to [0, 1] float
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_rgb = image_rgb.astype(np.float32) / 255.0

# Prepare list to hold transformed channels
fft_channels = []

# Apply DFT to each channel
for i in range(3):  # R, G, B
    channel = image_rgb[:, :, i]
    dft = cv2.dft(channel, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shifted = np.fft.fftshift(dft)

    # Compute magnitude and apply log scaling
    magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
    log_magnitude = np.log1p(magnitude)

    # Normalize to [0, 1] for display
    norm_magnitude = cv2.normalize(log_magnitude, None, 0, 1, cv2.NORM_MINMAX)
    fft_channels.append(norm_magnitude)

# Stack channels back into RGB image
fft_rgb = np.stack(fft_channels, axis=-1)

# Display using matplotlib
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(fft_rgb)
axes[1].set_title('DFT Magnitude Spectrum (RGB)')
axes[1].axis('off')

plt.tight_layout()
plt.show()
