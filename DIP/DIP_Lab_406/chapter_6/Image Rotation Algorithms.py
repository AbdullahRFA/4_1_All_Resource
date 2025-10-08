import numpy as np
import matplotlib.pyplot as plt
from skimage import io, transform, img_as_float

# Load grayscale image
image = img_as_float(io.imread('/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.03.tiff', as_gray=True))

# 1. Rotate by arbitrary angle (e.g., 60°)
rotate_nearest = transform.rotate(image, angle=60, order=0)   # Nearest neighbor
rotate_bicubic = transform.rotate(image, angle=60, order=3)   # Bicubic interpolation

# 2. Efficient rotation by 90°, 180°, 270°
rotate_90 = np.rot90(image, k=1)   # 90° counter-clockwise
rotate_180 = np.rot90(image, k=2)  # 180°
rotate_270 = np.rot90(image, k=3)  # 270°

# Display results
titles = [
    'Original', 'Rotate 60° Nearest', 'Rotate 60° Bicubic',
    'Rotate 90°', 'Rotate 180°', 'Rotate 270°'
]
images = [image, rotate_nearest, rotate_bicubic, rotate_90, rotate_180, rotate_270]

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, img, title in zip(axes.flatten(), images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
