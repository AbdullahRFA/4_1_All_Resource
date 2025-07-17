import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('lab.jpg', cv2.IMREAD_GRAYSCALE)
def add_gaussian_noise(image, mean=0, std=25):
    gauss = np.random.normal(mean, std, image.shape).astype('float32')
    noisy_image = image + gauss
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image
def add_rayleigh_noise(image, scale=25):
    rayleigh = np.random.rayleigh(scale, image.shape).astype('float32')
    noisy_image = image + rayleigh
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image
def add_erlang_noise(image, shape=2, scale=15):
    erlang = np.random.gamma(shape, scale, image.shape).astype('float32')
    noisy_image = image + erlang
    noisy_image = np.clip(noisy_image, 0, 255).astype('uint8')
    return noisy_image
gaussian_noisy_image = add_gaussian_noise(image)
rayleigh_noisy_image = add_rayleigh_noise(image)
erlang_noisy_image = add_erlang_noise(image)
plt.figure(figsize=(12, 8))
plt.subplot(2, 2, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.title("Gaussian Noise")
plt.imshow(gaussian_noisy_image, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 3)
plt.title("Rayleigh Noise")
plt.imshow(rayleigh_noisy_image, cmap='gray')
plt.axis('off')
plt.subplot(2, 2, 4)
plt.title("Erlang Noise")
plt.imshow(erlang_noisy_image, cmap='gray')
plt.axis('off')
plt.tight_layout()
plt.show()
