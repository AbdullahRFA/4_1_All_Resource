import matplotlib.pyplot as plt
from skimage import data, util
from scipy.ndimage import median_filter

# Load the iconic camera image
camera_image = data.camera()

# Normalize to [0, 1] for noise addition
camera_image_normalized = camera_image / 255.0

# Add salt and pepper noise
noisy_image = util.random_noise(camera_image_normalized, mode='s&p', amount=0.05)

# Apply median filters
median_3x3 = median_filter(noisy_image, size=3)
median_5x5 = median_filter(noisy_image, size=5)

# Plot results in 2x2 grid with display-friendly sizing
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

images = [
    (camera_image, 'Original Cameraman'),
    (noisy_image, 'Salt & Pepper Noise'),
    (median_3x3, 'Median Filter 3×3'),
    (median_5x5, 'Median Filter 5×5')
]

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()