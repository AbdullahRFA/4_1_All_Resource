import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert to RGB and normalize to [0, 1]
rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Step 2: Add salt and pepper noise
def add_salt_pepper_noise(img, amount=0.05):
    noisy = img.copy()
    total_pixels = img.shape[0] * img.shape[1]
    num_salt = int(amount * total_pixels / 2)
    num_pepper = int(amount * total_pixels / 2)

    for c in range(3):  # R, G, B channels
        # Salt
        coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape[:2]]
        noisy[coords[0], coords[1], c] = 1.0
        # Pepper
        coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape[:2]]
        noisy[coords[0], coords[1], c] = 0.0

    return noisy

noisy_image = add_salt_pepper_noise(rgb_image, amount=0.05)

# Step 3: Split into R, G, B channels
R_noisy, G_noisy, B_noisy = cv2.split(noisy_image)

# Step 4: Display noisy channels
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

# Step 5: Denoise each channel using OpenCV median filter
R_denoised = cv2.medianBlur((R_noisy * 255).astype(np.uint8), 5) / 255.0
G_denoised = cv2.medianBlur((G_noisy * 255).astype(np.uint8), 5) / 255.0
B_denoised = cv2.medianBlur((B_noisy * 255).astype(np.uint8), 5) / 255.0

# Step 6: Recombine denoised channels
denoised_image = cv2.merge([R_denoised, G_denoised, B_denoised])

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
