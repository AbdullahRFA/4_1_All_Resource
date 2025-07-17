import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('nature.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gray_image = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
laplacian_filtered_image = cv2.Laplacian(gray_image, cv2.CV_64F)
laplacian_filtered_image = np.uint8(np.absolute(laplacian_filtered_image))
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Grayscale Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(laplacian_filtered_image, cmap='gray')
plt.title('Laplacian Filtered Image')
plt.axis('off')
plt.show()
