import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('nature.jpeg')   
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)   
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)   
sobel_edge = np.sqrt(sobelx**2 + sobely**2)
plt.figure(figsize=(10,5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))   
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(sobel_edge, cmap='gray')
plt.title('Edge Detection using Sobel Operator')
plt.axis('off')
plt.show()
