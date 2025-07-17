import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms

# Load the input and reference images
input_image = cv2.imread('nature.jpeg', cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread('flower.jpeg', cv2.IMREAD_GRAYSCALE)

# Perform histogram matching
matched_image = match_histograms(input_image, reference_image)

# Display the input, reference, and matched images and their histograms
plt.figure(figsize=(12, 6))

# Show the input image
plt.subplot(2, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Show the reference image
plt.subplot(2, 3, 2)
plt.imshow(reference_image, cmap='gray')
plt.title('Reference Image')
plt.axis('off')

# Show the histogram-matched image
plt.subplot(2, 3, 3)
plt.imshow(matched_image, cmap='gray')
plt.title('Histogram-Matched Image')
plt.axis('off')

# Show the histogram of the input image
plt.subplot(2, 3, 4)
plt.hist(input_image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Input Image')

# Show the histogram of the reference image
plt.subplot(2, 3, 5)
plt.hist(reference_image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Reference Image')

# Show the histogram of the matched image
plt.subplot(2, 3, 6)
plt.hist(matched_image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Matched Image')

# Adjust the layout and show the figure
plt.tight_layout()
plt.show()
