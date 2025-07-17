import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load images
img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff")
img2 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.04.tiff")

# Show the original images
plt.imshow(img1)
plt.title("Image 1")
plt.axis('off')
plt.show()

plt.imshow(img2)
plt.title("Image 2")
plt.axis('off')
plt.show()

# Image Addition (commutative)
plt.imshow(img1 + img2)
plt.title("img1 + img2")
plt.axis('off')
plt.show()

plt.imshow(img2 + img1)
plt.title("img2 + img1")
plt.axis('off')
plt.show()

# Image Subtraction (not commutative)
plt.imshow(img1 - img2)
plt.title("img1 - img2")
plt.axis('off')
plt.show()

plt.imshow(img2 - img1)
plt.title("img2 - img1")
plt.axis('off')
plt.show()

# Flip vertically (up-down)
plt.imshow(np.flip(img1, 0))
plt.title("Flip Vertically (axis=0)")
plt.axis('off')
plt.show()

# Flip horizontally (left-right)
plt.imshow(np.flip(img1, 1))
plt.title("Flip Horizontally (axis=1)")
plt.axis('off')
plt.show()

# Roll the image (wrap around)
plt.imshow(np.roll(img1, 2048))
plt.title("Rolled Image")
plt.axis('off')
plt.show()

# Flip using fliplr (left to right)
plt.imshow(np.fliplr(img1))
plt.title("Flip Left-Right (fliplr)")
plt.axis('off')
plt.show()

# Flip using flipud (top to bottom)
plt.imshow(np.flipud(img1))
plt.title("Flip Up-Down (flipud)")
plt.axis('off')
plt.show()

# Rotate image 90 degrees counter-clockwise
plt.imshow(np.rot90(img1))
plt.title("Rotated 90°")
plt.axis('off')
plt.show()
