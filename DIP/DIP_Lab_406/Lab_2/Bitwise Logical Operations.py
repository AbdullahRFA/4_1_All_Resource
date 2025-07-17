import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load images
img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.07.tiff")
img2 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff")

# Ensure the images are in uint8 format for bitwise operations
if img1.dtype != np.uint8:
    img1 = (img1 * 255).astype(np.uint8)
if img2.dtype != np.uint8:
    img2 = (img2 * 255).astype(np.uint8)

# Bitwise AND
plt.imshow(np.bitwise_and(img1, img2))
plt.title("Bitwise AND")
plt.axis('off')
plt.show()

# Bitwise OR
plt.imshow(np.bitwise_or(img1, img2))
plt.title("Bitwise OR")
plt.axis('off')
plt.show()

# Bitwise XOR
plt.imshow(np.bitwise_xor(img1, img2))
plt.title("Bitwise XOR")
plt.axis('off')
plt.show()

# Bitwise NOT (Negative)
plt.subplot(1, 2, 1)
plt.imshow(img1)
plt.title("Original Image")
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(np.bitwise_not(img1))
plt.title("Bitwise NOT (Negative)")
plt.axis('off')

plt.tight_layout()
plt.show()
