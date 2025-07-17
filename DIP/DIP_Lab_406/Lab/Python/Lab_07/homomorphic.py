import cv2
import numpy as np
import matplotlib.pyplot as plt
def homomorphic_filter(input_image, low=0.5, high=2.0, D0=30):
    input_image = np.float32(input_image) / 255.0
    log_image = np.log1p(input_image)
    fft_image = np.fft.fft2(log_image)
    fft_shifted = np.fft.fftshift(fft_image)
    rows, cols = input_image.shape
    crow, ccol = rows // 2 , cols // 2  
    H = np.zeros((rows, cols), np.float32)
    for u in range(rows):
        for v in range(cols):
            D = np.sqrt((u - crow) ** 2 + (v - ccol) ** 2)
            H[u, v] = (high - low) * (1 - np.exp((-D ** 2) / (2 * (D0 ** 2)))) + low
    filtered_fft = fft_shifted * H
    ifft_shifted = np.fft.ifftshift(filtered_fft)
    inverse_fft = np.fft.ifft2(ifft_shifted)

    filtered_image = np.real(inverse_fft)
    homomorphic_image = np.expm1(filtered_image)

    homomorphic_image = cv2.normalize(homomorphic_image, None, 0, 1, cv2.NORM_MINMAX)

    return homomorphic_image

input_image = cv2.imread('lab.jpg', cv2.IMREAD_GRAYSCALE)

filtered_image = homomorphic_filter(input_image)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray')
plt.title('Homomorphic Filtered Image')
plt.axis('off')

plt.show()
