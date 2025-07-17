import numpy as np
import cv2
import matplotlib.pyplot as plt
def rgb_components_to_grayscale(image):
    image = image.astype(np.float32) / 255.0
    R = image[:, :, 0]
    G = image[:, :, 1]
    B = image[:, :, 2]
    grayscale_image = 0.2989 * R + 0.5870 * G + 0.1140 * B

    # Display the individual channels and grayscale image
    plt.figure(figsize=(10, 8))

    
    # Red component
    plt.subplot(3, 2, 3)
    plt.imshow(R, cmap='gray')
    plt.title('Red Component')
    plt.axis('off')

    # Green component
    plt.subplot(3, 2, 4)
    plt.imshow(G, cmap='gray')
    plt.title('Green Component')
    plt.axis('off')

    # Blue component
    plt.subplot(3, 2, 5)
    plt.imshow(B, cmap='gray')
    plt.title('Blue Component')
    plt.axis('off')

    # Grayscale image
    plt.subplot(3, 2, 6)
    plt.imshow(grayscale_image, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# Example usage:
image = cv2.imread('nature.jpeg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
rgb_components_to_grayscale(image)

