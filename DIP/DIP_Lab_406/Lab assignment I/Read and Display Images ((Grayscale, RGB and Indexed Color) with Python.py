import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans

# ----------- Step 1: Load Grayscale and RGB Images -----------
# Paths to your images
gray_img_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/5.3.01.tiff"
rgb_img_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff"

# Open and convert images properly
gray_img = Image.open(gray_img_path).convert('L')      # Grayscale
rgb_img = Image.open(rgb_img_path).convert('RGB')      # RGB

# Convert to NumPy arrays
gray_np = np.array(gray_img)
rgb_np = np.array(rgb_img)

# ----------- Step 2: Convert RGB to Indexed Color -----------
# Reshape RGB image to (num_pixels, 3)
pixels = rgb_np.reshape((-1, 3))

# Apply k-means clustering to quantize colors
n_colors = 16  # You can adjust this (e.g., 8, 16, 32)
kmeans = KMeans(n_clusters=n_colors, random_state=42)
labels = kmeans.fit_predict(pixels)
palette = kmeans.cluster_centers_.astype(np.uint8)

# Map each pixel to its closest palette color
indexed_img = palette[labels].reshape(rgb_np.shape)

# ----------- Step 3: Display All Images -----------
plt.figure(figsize=(18, 6))

# Grayscale
plt.subplot(1, 3, 1)
plt.imshow(gray_np, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')

# Original RGB
plt.subplot(1, 3, 2)
plt.imshow(rgb_np)
plt.title("Original RGB Image")
plt.axis('off')

# Indexed Color Image
plt.subplot(1, 3, 3)
plt.imshow(indexed_img)
plt.title(f"Indexed Color Image ({n_colors} colors)")
plt.axis('off')

plt.tight_layout()
plt.show()