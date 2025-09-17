import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.05.tiff'
gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if gray_image is None:
    raise FileNotFoundError("Image not found. Check the path.")

# --- Sobel Edge Detection ---
sobel_x = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
sobel_y = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
edges_sobel = cv2.magnitude(sobel_x, sobel_y)

# --- Prewitt Edge Detection ---
# Define Prewitt kernels
prewitt_kernel_x = np.array([[ -1, 0, 1],
                             [ -1, 0, 1],
                             [ -1, 0, 1]], dtype=np.float32)

prewitt_kernel_y = np.array([[ 1,  1,  1],
                             [ 0,  0,  0],
                             [-1, -1, -1]], dtype=np.float32)

prewitt_x = cv2.filter2D(gray_image, -1, prewitt_kernel_x)
prewitt_y = cv2.filter2D(gray_image, -1, prewitt_kernel_y)
edges_prewitt = cv2.magnitude(prewitt_x.astype(np.float32), prewitt_y.astype(np.float32))

# --- Roberts Edge Detection ---
# Define Roberts kernels
roberts_kernel_x = np.array([[1, 0],
                             [0, -1]], dtype=np.float32)

roberts_kernel_y = np.array([[0, 1],
                             [-1, 0]], dtype=np.float32)

roberts_x = cv2.filter2D(gray_image, -1, roberts_kernel_x)
roberts_y = cv2.filter2D(gray_image, -1, roberts_kernel_y)
edges_roberts = cv2.magnitude(roberts_x.astype(np.float32), roberts_y.astype(np.float32))

# Normalize for display
def normalize(img):
    return cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX)

# Plot results
fig, axes = plt.subplots(1, 4, figsize=(16, 5))
axes[0].imshow(gray_image, cmap='gray')
axes[0].set_title('Original Grayscale')
axes[0].axis('off')

axes[1].imshow(normalize(edges_prewitt), cmap='gray')
axes[1].set_title('Prewitt Edge')
axes[1].axis('off')

axes[2].imshow(normalize(edges_sobel), cmap='gray')
axes[2].set_title('Sobel Edge')
axes[2].axis('off')

axes[3].imshow(normalize(edges_roberts), cmap='gray')
axes[3].set_title('Roberts Edge')
axes[3].axis('off')

plt.tight_layout()
plt.show()
