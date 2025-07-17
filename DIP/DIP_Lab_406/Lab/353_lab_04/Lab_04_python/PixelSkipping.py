import cv2
import numpy as np
from matplotlib import pyplot as plt
image = cv2.imread('nature.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
skip_factor = 3  
skipped_image = image_rgb[::skip_factor, ::skip_factor]
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(skipped_image)
plt.title('Pixel Skipped Image')
plt.axis('off')
plt.show()
