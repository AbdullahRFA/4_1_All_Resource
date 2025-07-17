import cv2
import numpy as np
from matplotlib import pyplot as plt
from skimage.exposure import match_histograms

# Load the input and reference images
input_image = cv2.imread('nature.jpeg', cv2.IMREAD_GRAYSCALE)
reference_image = cv2.imread('flower.jpeg', cv2.IMREAD_GRAYSCALE)

# Check if the images are loaded successfully
if input_image is None:
    print("Error: Input image could not be loaded. Please check the file path.")
elif reference_image is None:
    print("Error: Reference image could not be loaded. Please check the file path.")
else:
    # Perform histogram matching without 'multichannel' argument
    matched_image = match_histograms(input_image, reference_image)

    # Plot the images and histograms
    plt.figure(figsize=(12, 6))
    
    # Display Input Image
    plt.subplot(2, 3, 1)
    plt.imshow(input_image, cmap='gray')
    plt.title('Input Image')
    plt.axis('off')
    
    # Display Reference Image
    plt.subplot(2, 3, 2)
    plt.imshow(reference_image, cmap='gray')
    plt.title('Reference Image')
    plt.axis('off')
    
    # Display Histogram-Matched Image
    plt.subplot(2, 3, 3)
    plt.imshow(matched_image, cmap='gray')
    plt.title('Histogram-Matched Image')
    plt.axis('off')
    
    # Plot histogram of Input Image
    plt.subplot(2, 3, 4)
    plt.hist(input_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Input Image')
    
    # Plot histogram of Reference Image
    plt.subplot(2, 3, 5)
    plt.hist(reference_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Reference Image')
    
    # Plot histogram of Matched Image
    plt.subplot(2, 3, 6)
    plt.hist(matched_image.ravel(), bins=256, range=[0, 256], color='black')
    plt.title('Histogram of Matched Image')
    
    # Display all plots
    plt.tight_layout()
    plt.show()
