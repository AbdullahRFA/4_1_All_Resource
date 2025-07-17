import cv2
import matplotlib.pyplot as plt

image = cv2.imread('nature.jpeg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)   

red_channel = image_rgb.copy()
green_channel = image_rgb.copy()
blue_channel = image_rgb.copy()

red_channel[:, :, 1] = 0   
red_channel[:, :, 2] = 0   

green_channel[:, :, 0] = 0   
green_channel[:, :, 2] = 0   

blue_channel[:, :, 0] = 0   
blue_channel[:, :, 1] = 0   

plt.figure(figsize=(12, 8))

plt.subplot(1, 4, 1)
plt.imshow(image_rgb)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 4, 2)
plt.imshow(red_channel)
plt.title("Red Channel")
plt.axis('off')

plt.subplot(1, 4, 3)
plt.imshow(green_channel)
plt.title("Green Channel")
plt.axis('off')

plt.subplot(1, 4, 4)
plt.imshow(blue_channel)
plt.title("Blue Channel")
plt.axis('off')

plt.tight_layout()
plt.show()

 
 