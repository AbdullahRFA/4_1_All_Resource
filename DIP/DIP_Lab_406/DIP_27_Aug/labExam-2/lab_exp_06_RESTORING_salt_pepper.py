import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load grayscale image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Replace with your own grayscale image.")
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Step 2: Add salt & pepper noise
def add_salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    total_pixels = img.size
    num_salt = int(amount * total_pixels / 2)
    num_pepper = int(amount * total_pixels / 2)

    # Salt noise
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 1.0

    # Pepper noise
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0.0

    return noisy

noisy_image = add_salt_pepper_noise(image, amount=0.05)

# Step 3: Convert to uint8 for OpenCV medianBlur
noisy_uint8 = (noisy_image * 255).astype(np.uint8)

# Step 4: Apply median filters
median_3x3 = cv2.medianBlur(noisy_uint8, 3)
median_5x5 = cv2.medianBlur(noisy_uint8, 5)

# Step 5: Display results
fig, axes = plt.subplots(2, 2, figsize=(10, 8), dpi=100)

images = [
    (image, 'Original Image'),
    (noisy_image, 'Salt & Pepper Noise'),
    (median_3x3 / 255.0, 'Median Filter 3×3'),
    (median_5x5 / 255.0, 'Median Filter 5×5')
]

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
