import cv2
import numpy as np
import matplotlib.pyplot as plt
input_image = cv2.imread('lab.jpg', cv2.IMREAD_GRAYSCALE)
kernel = np.ones((3, 3), np.uint8)  
eroded_image = cv2.erode(input_image, kernel, iterations=1)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(eroded_image, cmap='gray')
plt.title('Eroded Image')
plt.axis('off')
plt.show()
