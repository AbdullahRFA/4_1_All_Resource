import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import wiener
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

# Step 1: Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.06.tiff'

image_bgr = cv2.imread(image_path)  
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert to grayscale and normalize to [0, 1]
image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

# Step 2: Add Gaussian noise
noise = np.random.normal(0, np.sqrt(0.01), image_gray.shape)
noisy_image = np.clip(image_gray + noise, 0, 1)

# Step 3: Estimate signal and noise power
signal_power = np.var(image_gray)
noise_power = np.var(noisy_image - image_gray)
K = noise_power / signal_power

# Step 4: Apply Wiener filter with safe handling
def safe_wiener(img, mysize=(5, 5), noise=K):
    filtered = wiener(img, mysize=mysize, noise=noise)
    return np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)

restored_image = safe_wiener(noisy_image)

# Step 5: Evaluate restoration quality
psnr = peak_signal_noise_ratio(image_gray, restored_image, data_range=1.0)
ssim = structural_similarity(image_gray, restored_image, data_range=1.0)

# Step 6: Display results
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
axes[0].imshow(image_gray, cmap='gray')
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
