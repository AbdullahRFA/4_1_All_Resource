import cv2
import numpy as np
import matplotlib.pyplot as plt
image = cv2.imread('road.jpg')
inputImage = cv2.imread('road.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1.2,
    minDist=30,
    param1=50,
    param2=30,
    minRadius=10,
    maxRadius=80
)
if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])   
        radius = i[2]   
        cv2.circle(image, center, radius, (0, 255, 0), 3)   
        cv2.circle(image, center, 3, (0, 0, 255), 3)   
plt.figure(figsize=(10, 10))
plt.subplot(2, 2, 1)
plt.imshow(cv2.cvtColor(inputImage, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')
plt.subplot(2, 2, 2)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Detected Circles")
plt.axis('off')
plt.show()

