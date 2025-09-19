import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float, util
from skimage.filters import median
from skimage.morphology import disk

# Step 1: Load RGB image
rgb_image = img_as_float(data.astronaut())  # Normalized to [0, 1]

# Step 2: Add salt and pepper noise
noisy_image = util.random_noise(rgb_image, mode='s&p', amount=0.05)

# Step 3: Split into R, G, B noisy channels
R_noisy = noisy_image[:, :, 0]
G_noisy = noisy_image[:, :, 1]
B_noisy = noisy_image[:, :, 2]

# Step 4: Display noisy channels separately
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
axes[0].imshow(noisy_image)
axes[0].set_title('Noisy RGB Image')
axes[0].axis('off')

axes[1].imshow(R_noisy, cmap='gray')
axes[1].set_title('Noisy Red Channel')
axes[1].axis('off')

axes[2].imshow(G_noisy, cmap='gray')
axes[2].set_title('Noisy Green Channel')
axes[2].axis('off')

axes[3].imshow(B_noisy, cmap='gray')
axes[3].set_title('Noisy Blue Channel')
axes[3].axis('off')

plt.tight_layout()
plt.show()

# Step 5: Denoise each channel using median filtering
radius = 2  # You can adjust this for stronger smoothing
R_denoised = median(R_noisy, disk(radius))
G_denoised = median(G_noisy, disk(radius))
B_denoised = median(B_noisy, disk(radius))

# Step 6: Recombine denoised channels
denoised_image = np.stack((R_denoised, G_denoised, B_denoised), axis=2)

# Step 7: Display final result
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(rgb_image)
axes[0].set_title('Original RGB Image')
axes[0].axis('off')

axes[1].imshow(denoised_image)
axes[1].set_title('Denoised RGB Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()