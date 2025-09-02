import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.transform import resize

# Load sample grayscale image
camera_img = pywt.data.camera()

# Apply Discrete Wavelet Transform (DWT)
coeffs2 = pywt.dwt2(camera_img, 'db2')
cA, (cH, cV, cD) = coeffs2

# Combine subbands into one image (2x2 block)
rows, cols = cA.shape
combined = np.zeros((rows * 2, cols * 2), dtype=cA.dtype)

combined[0:rows, 0:cols] = cA       # Top-left
combined[0:rows, cols:] = cH        # Top-right
combined[rows:, 0:cols] = cV        # Bottom-left
combined[rows:, cols:] = cD         # Bottom-right

# Define white value based on image type
white_val = 255 if np.issubdtype(combined.dtype, np.integer) else 1.0

# Function to draw 3-pixel-wide border around a subband
def draw_border(img, top, left, height, width, thickness=3, color=255):
    # Top
    img[top:top+thickness, left:left+width] = color
    # Bottom
    img[top+height-thickness:top+height, left:left+width] = color
    # Left
    img[top:top+height, left:left+thickness] = color
    # Right
    img[top:top+height, left+width-thickness:left+width] = color

# Draw white borders around all four subbands
draw_border(combined, 0, 0, rows, cols, thickness=3, color=white_val)                # cA
draw_border(combined, 0, cols, rows, cols, thickness=3, color=white_val)             # cH
draw_border(combined, rows, 0, rows, cols, thickness=3, color=white_val)             # cV
draw_border(combined, rows, cols, rows, cols, thickness=3, color=white_val)          # cD

# Resize original image to match combined layout
camera_resized = resize(camera_img, combined.shape, preserve_range=True, anti_aliasing=True).astype(combined.dtype)

# Display side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
titles = ['Original Image (Resized)', 'Wavelet Subbands with White Borders']
images = [camera_resized, combined]

for ax, title, img in zip(axs, titles, images):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
