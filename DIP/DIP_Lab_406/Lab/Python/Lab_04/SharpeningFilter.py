import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('nature.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])
sharpened_image = cv2.filter2D(image_rgb, -1, sharpening_kernel)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sharpened_image)
plt.title('Sharpened Image')
plt.axis('off')
plt.show()
