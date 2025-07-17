import cv2
import numpy as np
from matplotlib import pyplot as plt

input_image = cv2.imread('nature.jpeg', cv2.IMREAD_GRAYSCALE)   

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(input_image.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Input Image')

equalized_image = cv2.equalizeHist(input_image)

plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram-Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()
import cv2
import numpy as np
from matplotlib import pyplot as plt

input_image = cv2.imread('your_image.jpg', cv2.IMREAD_GRAYSCALE)   

plt.figure(figsize=(10, 8))

plt.subplot(2, 2, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.hist(input_image.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Input Image')

equalized_image = cv2.equalizeHist(input_image)

plt.subplot(2, 2, 3)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram-Equalized Image')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.hist(equalized_image.ravel(), bins=256, range=[0, 256])
plt.title('Histogram of Equalized Image')

plt.tight_layout()
plt.show()