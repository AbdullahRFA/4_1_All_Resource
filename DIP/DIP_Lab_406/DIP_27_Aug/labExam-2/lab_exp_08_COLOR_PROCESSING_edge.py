import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert to RGB and normalize to [0, 1]
rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Step 2: Split into R, G, B channels
R, G, B = cv2.split(rgb_image)

# Step 3: Apply Sobel edge detection to each channel
def sobel_edge(channel):
    dx = cv2.Sobel(channel, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(channel, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(dx, dy)
    return cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)

R_edge = sobel_edge(R)
G_edge = sobel_edge(G)
B_edge = sobel_edge(B)

# Step 4: Recombine edge-detected channels into RGB
edge_rgb = cv2.merge([R_edge, G_edge, B_edge])

# Step 5: Create grayscale edge map
edge_gray = (R_edge + G_edge + B_edge) / 3

# Step 6: Display results
fig, axes = plt.subplots(1, 6, figsize=(24, 5))

images = [
    (rgb_image, 'Original RGB'),
    (R_edge, 'Red Channel Edges'),
    (G_edge, 'Green Channel Edges'),
    (B_edge, 'Blue Channel Edges'),
    (edge_rgb, 'Combined Edge RGB'),
    (edge_gray, 'Grayscale Edge Map')
]

for ax, (img, title) in zip(axes, images):
    ax.imshow(img, cmap='gray' if img.ndim == 2 else None)
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
