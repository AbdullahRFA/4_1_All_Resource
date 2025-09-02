import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from skimage import data, color, util
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Load a sample image and convert to grayscale
image = color.rgb2gray(data.astronaut())

# Add Gaussian noise
noisy_image = util.random_noise(image, mode='gaussian', var=0.01)

# Estimate signal and noise power
signal_power = np.var(image)
noise_power = np.var(noisy_image - image)
K = noise_power / signal_power  # Noise-to-signal power ratio

# Apply Wiener filter with safe handling
def safe_wiener(img, mysize=(5, 5), noise=K):
    filtered = wiener(img, mysize=mysize, noise=noise)
    # Replace NaNs and infs with zeros
    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    return filtered

# Restore image
restored_image = safe_wiener(noisy_image)

# Evaluate restoration quality
psnr = peak_signal_noise_ratio(image, restored_image, data_range=1.0)
ssim = structural_similarity(image, restored_image, data_range=1.0)

# Display results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(noisy_image, cmap='gray')
axes[1].set_title('Noisy Image')
axes[1].axis('off')

axes[2].imshow(restored_image, cmap='gray')
axes[2].set_title(f'Restored\nPSNR: {psnr:.2f}, SSIM: {ssim:.2f}, K: {K:.4f}')
axes[2].axis('off')

plt.tight_layout()
plt.show()