import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('nature.jpeg', 0)  
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
def notch_filter(dft_shift, center, radius=10):
    rows, cols = dft_shift.shape
    crow, ccol = int(rows / 2), int(cols / 2)   
    mask = np.ones((rows, cols), np.uint8)
    cv2.circle(mask, center, radius, 0, thickness=-1)   
    fshift = dft_shift * mask
    return fshift
filtered_dft = notch_filter(dft_shift, (130, 130), radius=15)
filtered_dft = notch_filter(filtered_dft, (170, 170), radius=15)   
f_ishift = np.fft.ifftshift(filtered_dft)
img_back = np.fft.ifft2(f_ishift)
img_back = np.abs(img_back)
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray'), plt.title('Original Image')
plt.subplot(1, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum')
plt.subplot(1, 3, 3), plt.imshow(img_back, cmap='gray'), plt.title('Filtered Image')
plt.show()