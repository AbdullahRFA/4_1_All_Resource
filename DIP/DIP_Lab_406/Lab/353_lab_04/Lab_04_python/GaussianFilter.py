import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('nature.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
gaussian_filtered_image = cv2.GaussianBlur(image_rgb, (7, 7), 0)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(gaussian_filtered_image)
plt.title('Gaussian Filtered Image')
plt.axis('off')
plt.show()
