import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert BGR to RGB and normalize to [0, 1]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width, _ = image_rgb.shape

# Step 2: Define center
center_x, center_y = width // 2, height // 2

# Step 3: Compute distance of each pixel from center
Y, X = np.indices((height, width))
distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

# Step 4: Define radial band size
band_size = 10
max_radius = int(np.max(distances))
num_bands = max_radius // band_size + 1

# Step 5: Create output image
radial_pixelized = np.zeros_like(image_rgb)

# Step 6: Apply pixelization per radial band
for band in range(num_bands):
    r_min = band * band_size
    r_max = (band + 1) * band_size
    mask = (distances >= r_min) & (distances < r_max)

    for c in range(3):  # R, G, B channels
        avg_color = np.mean(image_rgb[:, :, c][mask])
        radial_pixelized[:, :, c][mask] = avg_color

# Step 7: Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(radial_pixelized)
axes[1].set_title('Radial Pixelized Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()

















"""
Great question 👍 Let’s carefully break down your code step by step.

This program applies a radial pixelization effect on an image — meaning it divides the image into circular rings (bands) around the center and assigns each band an average color.

⸻

📌 Step-by-step Explanation

Step 1: Load an RGB image

image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

	•	cv2.imread() loads the image in BGR format.
	•	Converted to RGB (since matplotlib expects RGB).
	•	Normalized to [0, 1] for easier averaging.
	•	height, width, _ = image_rgb.shape stores image dimensions.

⸻

Step 2: Define the image center

center_x, center_y = width // 2, height // 2

	•	Finds the center pixel of the image (acts as the origin for radial distances).

⸻

Step 3: Compute distances from the center

Y, X = np.indices((height, width))
distances = np.sqrt((X - center_x)**2 + (Y - center_y)**2)

	•	np.indices() creates coordinate grids (X, Y).
	•	Computes Euclidean distance of each pixel from the center.
	•	So distances[y, x] tells how far pixel (x, y) is from the center.

⸻

Step 4: Define radial bands

band_size = 10
max_radius = int(np.max(distances))
num_bands = max_radius // band_size + 1

	•	band_size = 10 → pixels are grouped into circular rings 10 pixels wide.
	•	max_radius → distance of farthest pixel from center.
	•	num_bands → total number of circular rings.

⸻

Step 5: Initialize output image

radial_pixelized = np.zeros_like(image_rgb)

	•	Creates an empty image with same size as input.
	•	Will store the pixelized (radial-averaged) version.

⸻

Step 6: Pixelization per radial band

for band in range(num_bands):
    r_min = band * band_size
    r_max = (band + 1) * band_size
    mask = (distances >= r_min) & (distances < r_max)

    for c in range(3):  # R, G, B channels
        avg_color = np.mean(image_rgb[:, :, c][mask])
        radial_pixelized[:, :, c][mask] = avg_color

	•	For each radial band:
	•	mask selects all pixels whose distance is between r_min and r_max.
	•	For each channel (R, G, B):
	•	Compute average intensity of that channel within the ring.
	•	Assign the average back to all pixels in that ring.
	•	Effect: Each circular ring gets a uniform color → pixelization effect.

⸻

Step 7: Display results

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)              # Original
axes[1].imshow(radial_pixelized)       # Radial Pixelized

	•	Plots original and pixelized images side by side.

⸻

🎨 Effect on Image
	•	Instead of a smooth gradient of colors, each circular ring has a single averaged color.
	•	This creates a radial “posterization” or pixelation effect, where the image looks like it’s made of concentric colored rings.

⸻

✅ In short:
This code takes an image, divides it into concentric circular rings around the center, and replaces each ring with its average color. That’s why it’s called Radial Pixelization.

⸻

👉 Do you want me to also show you how to modify this code for square-block pixelization (like Minecraft effect) instead of radial rings?
"""