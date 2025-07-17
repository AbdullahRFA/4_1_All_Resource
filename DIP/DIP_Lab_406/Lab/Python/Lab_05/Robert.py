import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('lab_05.jpg')   
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)   
roberts_x = np.array([[1, 0],
                      [0, -1]])  
roberts_y = np.array([[0, 1],
                      [-1, 0]])   
edge_x = cv2.filter2D(gray, -1, roberts_x)
edge_y = cv2.filter2D(gray, -1, roberts_y)
roberts_edge = np.sqrt(edge_x**2 + edge_y**2)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(roberts_edge, cmap='gray')
plt.title('Edge Detection using Roberts Operator')
plt.axis('off')
plt.show()
