import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('lab_05.jpg')  
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
isotropic_x = np.array([[-1, 0, 1],
                        [-np.sqrt(2), 0, np.sqrt(2)],
                        [-1, 0, 1]])   

isotropic_y = np.array([[-1, -np.sqrt(2), -1],
                        [ 0, 0, 0],
                        [ 1, np.sqrt(2), 1]])   
edge_x = cv2.filter2D(gray, -1, isotropic_x)
edge_y = cv2.filter2D(gray, -1, isotropic_y)
isotropic_edge = np.sqrt(edge_x**2 + edge_y**2)
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))   
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(isotropic_edge, cmap='gray')
plt.title('Edge Detection using Isotropic Operator')
plt.axis('off')
plt.show()
