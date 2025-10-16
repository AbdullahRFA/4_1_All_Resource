import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert to RGB and normalize to [0, 1]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width, _ = image_rgb.shape

# Ripple parameters
amplitude = 5       # pixels
frequency = 0.05    # ripple density
phase = 0           # phase shift
center_x, center_y = width // 2, height // 2

# Generate coordinate grid
Y, X = np.indices((height, width), dtype=np.float32)
dx = X - center_x
dy = Y - center_y
r = np.sqrt(dx**2 + dy**2)

offset = amplitude * np.sin(frequency * r + phase)

# Avoid division by zero
r_safe = r + 1e-6

X_new = X + (dx / r_safe) * offset
Y_new = Y + (dy / r_safe) * offset

# Remap using OpenCV
map_x = X_new.astype(np.float32)
map_y = Y_new.astype(np.float32)
rippled_bgr = cv2.remap(image_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
rippled_rgb = cv2.cvtColor(rippled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

# Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(rippled_rgb)
axes[1].set_title('Ripple Effect')
axes[1].axis('off')

plt.tight_layout()
plt.show()















"""
Perfect 👍 This code applies a Ripple Effect (water wave distortion) on an image. Let’s go step by step so you clearly understand what’s happening.

⸻

📌 Step-by-Step Explanation

Step 1: Load Image

image_bgr = cv2.imread(image_path)  
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width, _ = image_rgb.shape

	•	Load the image in BGR format (OpenCV default).
	•	Convert it to RGB (for matplotlib display).
	•	Normalize pixel values to [0, 1].
	•	Store height and width for later use.

⸻

Step 2: Define Ripple Parameters

amplitude = 5       # how strong (height) the ripples are
frequency = 0.05    # how dense (close together) the ripples are
phase = 0           # starting shift of ripples
center_x, center_y = width // 2, height // 2

	•	amplitude → Controls how far pixels are displaced (bigger = stronger ripple).
	•	frequency → Controls how many ripples per distance (bigger = more ripples).
	•	phase → Shifts ripples left or right (like moving the wave).
	•	center_x, center_y → Ripples are centered at the middle of the image.

⸻

Step 3: Compute Pixel Distances

Y, X = np.indices((height, width), dtype=np.float32)
dx = X - center_x
dy = Y - center_y
r = np.sqrt(dx**2 + dy**2)

	•	np.indices gives grids of X and Y coordinates.
	•	dx, dy are displacements from the image center.
	•	r is the distance of each pixel from the center (like radius in polar coordinates).

⸻

Step 4: Ripple Distortion

offset = amplitude * np.sin(frequency * r + phase)

	•	For each pixel, calculate how much to shift outward or inward using a sine wave.
	•	sin(frequency * r) creates circular ripples from the center.
	•	Multiplying by amplitude scales the ripple strength.

⸻

Step 5: Apply Offset

r_safe = r + 1e-6
X_new = X + (dx / r_safe) * offset
Y_new = Y + (dy / r_safe) * offset

	•	Avoid dividing by zero by adding a small 1e-6.
	•	(dx / r_safe, dy / r_safe) → unit vector pointing radially outward.
	•	Multiply by offset to move pixels inwards or outwards along this direction.
	•	X_new, Y_new are the new distorted coordinates for each pixel.

⸻

Step 6: Remap Image

map_x = X_new.astype(np.float32)
map_y = Y_new.astype(np.float32)
rippled_bgr = cv2.remap(image_bgr, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
rippled_rgb = cv2.cvtColor(rippled_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

	•	cv2.remap takes the original image and moves pixels to new positions (X_new, Y_new).
	•	interpolation=cv2.INTER_LINEAR → smooth interpolation.
	•	borderMode=cv2.BORDER_REFLECT → mirror effect at edges to avoid black gaps.
	•	Convert back to RGB and normalize.

⸻

Step 7: Show Images

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)         # original
axes[1].imshow(rippled_rgb)       # with ripple effect

	•	Display original image and ripple-transformed image side by side.

⸻

🎨 What It Does to the Image
	•	The ripple effect creates concentric circular waves radiating from the image center.
	•	Pixels shift outward/inward depending on the sine wave, making the image look like it’s under water or being distorted by waves.

⸻

✅ In short:
This code uses polar coordinates (distance from center) + sine function to distort pixels in a circular ripple pattern, then remaps the image to create a water ripple effect.

⸻

👉 Do you want me to also modify this ripple code so you can control ripple direction (horizontal/vertical waves instead of circular)?
"""