import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the input image
input_image = cv2.imread('nature.jpeg')

# Convert the image to grayscale if it is in RGB
if len(input_image.shape) == 3:
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

# Function to compute the CDF
def compute_cdf(image):
    hist, bins = np.histogram(image.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf / cdf.max()  # Normalize to [0, 1]
    return cdf_normalized

# Display the input image and its histogram
plt.figure(figsize=(12, 6))

# Show input image
plt.subplot(2, 3, 1)
plt.imshow(input_image, cmap='gray')
plt.title('Input Image')
plt.axis('off')

# Plot histogram of the input image
plt.subplot(2, 3, 2)
plt.hist(input_image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Input Image')

# Compute and plot CDF of the input image
cdf_input = compute_cdf(input_image)

plt.subplot(2, 3, 3)
plt.plot(cdf_input, color='black', linewidth=2)
plt.title('CDF of Input Image')
plt.xlabel('Pixel Intensity Values')
plt.ylabel('CDF')

# Perform histogram equalization
equalized_image = cv2.equalizeHist(input_image)

# Display the equalized image and its histogram
plt.subplot(2, 3, 4)
plt.imshow(equalized_image, cmap='gray')
plt.title('Histogram-Equalized Image')
plt.axis('off')

# Plot histogram of the equalized image
plt.subplot(2, 3, 5)
plt.hist(equalized_image.ravel(), bins=256, range=[0, 256], color='black')
plt.title('Histogram of Equalized Image')

# Compute and plot CDF of the equalized image
cdf_equalized = compute_cdf(equalized_image)

plt.subplot(2, 3, 6)
plt.plot(cdf_equalized, color='black', linewidth=2)
plt.title('CDF of Equalized Image')
plt.xlabel('Pixel Intensity Values')
plt.ylabel('CDF')

plt.tight_layout()
plt.show()
