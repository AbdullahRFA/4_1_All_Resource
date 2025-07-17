import cv2
from matplotlib import pyplot as plt
image = cv2.imread('nature.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
new_width, new_height = 400, 300 
resized_image = cv2.resize(image_rgb, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')
plt.subplot(1, 2, 2)
plt.imshow(resized_image)
plt.title('Resized Image (Linear Interpolation)')
plt.axis('off')
plt.show()
