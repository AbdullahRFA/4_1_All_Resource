import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image = cv2.imread('nature.jpeg', cv2.IMREAD_GRAYSCALE)
fft_image = np.fft.fft2(input_image)
fft_shifted = np.fft.fftshift(fft_image)
magnitude_spectrum = 20 * np.log(np.abs(fft_shifted) + 1)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (FFT)')
plt.axis('off')
plt.show()
