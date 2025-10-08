import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import io, transform, img_as_float

# Load grayscale image
image = img_as_float(io.imread('/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.05.tiff', as_gray=True))

# 1. Interpolation-Based Scaling
nearest = transform.rescale(image, scale=2, order=0, anti_aliasing=False)
bilinear = transform.rescale(image, scale=2, order=1, anti_aliasing=True)
bicubic = transform.rescale(image, scale=2, order=3, anti_aliasing=True)

# 2. Zero-Interleaving + Spatial Filtering
def zero_interleave(img):
    r, c = img.shape
    interleaved = np.zeros((2*r, 2*c))
    interleaved[::2, ::2] = img
    return interleaved

hz = zero_interleave(image)

# Define filters
nearest_filter = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]])
bilinear_filter = np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]]) / 4.0
bicubic_filter = np.array([
    [1, 4, 6, 4, 1],
    [4, 16, 24, 16, 4],
    [6, 24, 36, 24, 6],
    [4, 16, 24, 16, 4],
    [1, 4, 6, 4, 1]
]) / 64.0

# Apply filters
hz_nearest = convolve(hz, nearest_filter, mode='constant')
hz_bilinear = convolve(hz, bilinear_filter, mode='constant')
hz_bicubic = convolve(hz, bicubic_filter, mode='constant')

# 3. Image Minimization (Subsampling)
minimized_nearest = image[::4, ::4]
minimized_bicubic = transform.rescale(image, scale=0.25, order=3, anti_aliasing=True)

# Display results
titles = [
    'Original', 'Nearest Interpolation', 'Bilinear Interpolation', 'Bicubic Interpolation',
    'Zero-Interleaved Nearest', 'Zero-Interleaved Bilinear', 'Zero-Interleaved Bicubic',
    'Minimized Nearest', 'Minimized Bicubic'
]
images = [
    image, nearest, bilinear, bicubic,
    hz_nearest, hz_bilinear, hz_bicubic,
    minimized_nearest, minimized_bicubic
]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
for ax, img, title in zip(axes.flatten(), images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
