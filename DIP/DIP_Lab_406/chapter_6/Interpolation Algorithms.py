import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_float

# Load grayscale image
image = img_as_float(io.imread('/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff', as_gray=True))

# Resize factors
scale_factor = 2  # Enlarge image by 2x

# 1. Nearest Neighbor Interpolation
nearest = transform.rescale(image, scale=scale_factor, order=0, anti_aliasing=False)

# 2. Bilinear Interpolation
bilinear = transform.rescale(image, scale=scale_factor, order=1, anti_aliasing=True)

# 3. Bicubic Interpolation
bicubic = transform.rescale(image, scale=scale_factor, order=3, anti_aliasing=True)

# Display results
titles = ['Original', 'Nearest Neighbor', 'Bilinear', 'Bicubic']
images = [image, nearest, bilinear, bicubic]

fig, axes = plt.subplots(1, 4, figsize=(20, 5))
for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
