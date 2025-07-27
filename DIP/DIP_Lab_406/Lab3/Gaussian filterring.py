import matplotlib.pyplot as plt              # For image plotting
import matplotlib.image as mpimg             # For reading image files
import numpy as np                           # For numerical operations
import cv2                                   # OpenCV for image processing

# Step 1: Read RGB image
img = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.06.tiff")

# Step 2: Convert RGB image to Grayscale using OpenCV
# OpenCV uses BGR, so we convert from RGB to BGR and then to GRAY
img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Step 3: Apply 5x5 Gaussian filter with σ = 1
gaussian_1 = cv2.GaussianBlur(gray, (5, 5), sigmaX=1)

# Step 4: Apply 5x5 Gaussian filter with σ = 3
gaussian_3 = cv2.GaussianBlur(gray, (5, 5), sigmaX=3)

# Display all images
titles = ['Original RGB Image', 'Grayscale Image', 'Gaussian σ=1', 'Gaussian σ=3']
images = [img, gray, gaussian_1, gaussian_3]

plt.figure(figsize=(10, 8))
for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.axis('off')
    plt.title(titles[i])
    if i == 0:
        plt.imshow(images[i])               # Show RGB image
    else:
        plt.imshow(images[i], cmap='gray')  # Show grayscale images

plt.tight_layout()
plt.show()
