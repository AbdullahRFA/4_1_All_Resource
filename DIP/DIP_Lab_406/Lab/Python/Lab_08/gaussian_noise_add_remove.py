import cv2
import numpy as np
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, var=0.01):
    row, col = image.shape
    sigma = var ** 0.5
    gaussian = np.random.normal(mean, sigma, (row, col))
    noisy_image = image + gaussian * 255
    return np.clip(noisy_image, 0, 255).astype(np.uint8)

def arithmetic_mean_filter(image, kernel_size=3):
    return cv2.blur(image, (kernel_size, kernel_size))

def geometric_mean_filter(image, kernel_size=3):
    image_float = image.astype(np.float32)
    img_log = np.log1p(image_float)
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    log_filtered = cv2.filter2D(img_log, -1, kernel)
    geometric_mean_img = np.expm1(log_filtered)
    return np.clip(geometric_mean_img, 0, 255).astype(np.uint8)

# Load the image in grayscale
image = cv2.imread('lab.jpg', 0)

# Add Gaussian noise to the image
noisy_image = add_gaussian_noise(image)

# Apply arithmetic mean filter to the noisy image
arithmetic_filtered_image = arithmetic_mean_filter(noisy_image)

# Apply geometric mean filter to the noisy image
geometric_filtered_image = geometric_mean_filter(noisy_image)

# Display the images
plt.figure(figsize=(10, 5))
plt.subplot(1, 4, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 4, 2), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy (Gaussian)')
plt.subplot(1, 4, 3), plt.imshow(arithmetic_filtered_image, cmap='gray'), plt.title('Arithmetic Mean Filter')
plt.subplot(1, 4, 4), plt.imshow(geometric_filtered_image, cmap='gray'), plt.title('Geometric Mean Filter')
plt.show()
