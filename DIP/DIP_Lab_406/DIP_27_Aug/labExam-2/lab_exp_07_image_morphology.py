import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load grayscale image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Replace with your own grayscale image.")

# Step 2: Apply Otsu's thresholding
_, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Step 3: Define structuring element (disk-shaped approximation)
selem = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))  # disk radius ≈ 3

# Step 4: Apply morphological operations
eroded = cv2.erode(binary_image, selem)
dilated = cv2.dilate(binary_image, selem)

# Step 5: Display results
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

images = [
    (image, 'Original Grayscale Image'),
    (binary_image, 'Binary Image (Otsu Thresholded)'),
    (eroded, 'Eroded Image'),
    (dilated, 'Dilated Image')
]

for ax, (img, title) in zip(axes.flat, images):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
