import numpy as np
import matplotlib.pyplot as plt
import pywt
from skimage.data import astronaut
from skimage.transform import resize

# Load sample RGB image
rgb_img = astronaut()

# Apply DWT to each channel separately
coeffs_r = pywt.dwt2(rgb_img[:, :, 0], 'db2')
coeffs_g = pywt.dwt2(rgb_img[:, :, 1], 'db2')
coeffs_b = pywt.dwt2(rgb_img[:, :, 2], 'db2')

# Unpack coefficients
cA_r, (cH_r, cV_r, cD_r) = coeffs_r
cA_g, (cH_g, cV_g, cD_g) = coeffs_g
cA_b, (cH_b, cV_b, cD_b) = coeffs_b

# Function to combine subbands into one image
def combine_subbands(cA, cH, cV, cD):
    rows, cols = cA.shape
    combined = np.zeros((rows * 2, cols * 2), dtype=cA.dtype)
    combined[0:rows, 0:cols] = cA
    combined[0:rows, cols:] = cH
    combined[rows:, 0:cols] = cV
    combined[rows:, cols:] = cD
    return combined

# Function to normalize image for display
def normalize(img):
    img_min, img_max = np.min(img), np.max(img)
    return ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)

# Combine and normalize subbands for each channel
combined_r = normalize(combine_subbands(cA_r, cH_r, cV_r, cD_r))
combined_g = normalize(combine_subbands(cA_g, cH_g, cV_g, cD_g))
combined_b = normalize(combine_subbands(cA_b, cH_b, cV_b, cD_b))

# Stack channels back into RGB image
combined_rgb = np.stack([combined_r, combined_g, combined_b], axis=-1)

# Function to draw border on RGB image
def draw_border_rgb(img, top, left, height, width, thickness=3, color=[255, 255, 255]):
    for ch in range(3):
        img[top:top+thickness, left:left+width, ch] = color[ch]
        img[top+height-thickness:top+height, left:left+width, ch] = color[ch]
        img[top:top+height, left:left+thickness, ch] = color[ch]
        img[top:top+height, left+width-thickness:left+width, ch] = color[ch]

# Draw borders around each subband block
rows, cols = cA_r.shape
white_val = [255, 255, 255]
draw_border_rgb(combined_rgb, 0, 0, rows, cols, color=white_val)                # cA
draw_border_rgb(combined_rgb, 0, cols, rows, cols, color=white_val)             # cH
draw_border_rgb(combined_rgb, rows, 0, rows, cols, color=white_val)             # cV
draw_border_rgb(combined_rgb, rows, cols, rows, cols, color=white_val)          # cD

# Resize original image to match combined layout
resized_rgb = resize(rgb_img, combined_rgb.shape, preserve_range=True, anti_aliasing=True).astype(np.uint8)

# Display side-by-side
fig, axs = plt.subplots(1, 2, figsize=(14, 6))
titles = ['Original RGB Image (Resized)', 'Wavelet Subbands with White Borders']
images = [resized_rgb, combined_rgb]

for ax, title, img in zip(axs, titles, images):
    ax.imshow(img)
    ax.set_title(title)
    ax.axis('off')

# Optional: Add quadrant labels to subbands
labels = ['cA', 'cH', 'cV', 'cD']
positions = [(rows//2, cols//2), (rows//2, cols + cols//2),
             (rows + rows//2, cols//2), (rows + rows//2, cols + cols//2)]

for label, (y, x) in zip(labels, positions):
    axs[1].text(x, y, label, color='white', fontsize=12, ha='center', va='center')

plt.tight_layout()
plt.show()