import cv2
import numpy as np
import matplotlib.pyplot as plt
def add_salt_and_pepper_noise(image, salt_prob=0.05, pepper_prob=0.05):
    noisy_image = np.copy(image)
    num_salt = np.ceil(salt_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 255
    num_pepper = np.ceil(pepper_prob * image.size)
    coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
    noisy_image[coords[0], coords[1]] = 0
    return noisy_image
def remove_salt_and_pepper_noise(image, kernel_size=3):
    denoised_image = cv2.medianBlur(image, kernel_size)
    return denoised_image
image = cv2.imread('nature.jpeg', 0)   
noisy_image = add_salt_and_pepper_noise(image)
denoised_image = remove_salt_and_pepper_noise(noisy_image)
plt.figure(figsize=(10, 5))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(noisy_image, cmap='gray'), plt.title('Noisy (Salt & Pepper)')
plt.subplot(1, 3, 3), plt.imshow(denoised_image, cmap='gray'), plt.title('Denoised (Median Filter)')
plt.show()