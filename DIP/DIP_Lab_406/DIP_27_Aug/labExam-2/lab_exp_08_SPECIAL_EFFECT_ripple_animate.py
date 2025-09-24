import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Load image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert to RGB and normalize to [0, 1]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width, _ = image_rgb.shape

# Ripple parameters
base_amplitude = 5
frequency = 0.2
center_x, center_y = width // 2, height // 2

# Precompute coordinate grid
Y, X = np.indices((height, width), dtype=np.float32)
dx = X - center_x
dy = Y - center_y
r = np.sqrt(dx**2 + dy**2) + 1e-6  # Avoid division by zero

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(image_rgb)
ax.axis('off')

def update(frame):
    time = frame * 0.3
    wavefront_radius = time * 50
    mask = r < wavefront_radius
    edge_fade = np.clip((wavefront_radius - r) / 20, 0, 1)
    wave = np.sin(frequency * r - time)
    amplitude = base_amplitude * edge_fade
    offset = amplitude * wave * mask

    X_new = X + (dx / r) * offset
    Y_new = Y + (dy / r) * offset

    # Remap each channel
    remapped = np.zeros_like(image_rgb)
    for c in range(3):
        channel = (image_rgb[:, :, c] * 255).astype(np.uint8)
        warped = cv2.remap(channel, X_new.astype(np.float32), Y_new.astype(np.float32),
                           interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        remapped[:, :, c] = warped.astype(np.float32) / 255.0

    im.set_data(remapped)
    return [im]

# Create animation
ani = FuncAnimation(fig, update, frames=80, interval=100, blit=True)
plt.show()





















"""
Great — this one is really cool because it animates a propagating ripple effect (like dropping a stone into water and watching waves spread). Let’s break it down clearly:

⸻

📌 Step-by-Step Explanation

Step 1: Load Image

image_bgr = cv2.imread(image_path)  
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width, _ = image_rgb.shape

	•	Load image with OpenCV (BGR).
	•	Convert to RGB for plotting and normalize pixel values to [0,1].
	•	Extract height and width.

⸻

Step 2: Define Ripple Parameters

base_amplitude = 5      # how strong the ripples are
frequency = 0.2         # density of ripples
center_x, center_y = width // 2, height // 2

	•	base_amplitude → Maximum displacement of pixels.
	•	frequency → Controls how many ripple cycles per distance.
	•	center_x, center_y → Ripples originate from image center.

⸻

Step 3: Precompute Coordinate Grid

Y, X = np.indices((height, width), dtype=np.float32)
dx = X - center_x
dy = Y - center_y
r = np.sqrt(dx**2 + dy**2) + 1e-6

	•	np.indices gives grid of pixel coordinates.
	•	dx, dy = displacement from image center.
	•	r = radial distance from center (like radius).
	•	+1e-6 avoids division by zero when pixel = center.

⸻

Step 4: Setup Animation

fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(image_rgb)
ax.axis('off')

	•	Use matplotlib.animation.FuncAnimation for animation.
	•	Show the initial image (imshow).
	•	Turn off axis.

⸻

Step 5: Update Function (Runs Every Frame)

def update(frame):
    time = frame * 0.3
    wavefront_radius = time * 50

	•	frame increases with each animation step.
	•	time = scaled frame number → controls wave progression.
	•	wavefront_radius = how far the ripple has propagated outward.

⸻

Wave Physics

mask = r < wavefront_radius
edge_fade = np.clip((wavefront_radius - r) / 20, 0, 1)
wave = np.sin(frequency * r - time)
amplitude = base_amplitude * edge_fade
offset = amplitude * wave * mask

	•	mask → Only apply ripples inside the expanding wavefront.
	•	edge_fade → Makes ripples fade smoothly at the edge.
	•	wave = sine wave based on distance r and time.
	•	amplitude = decreases near wavefront to avoid sharp cutoff.
	•	offset = final displacement strength for each pixel.

⸻

Apply Ripple Distortion

X_new = X + (dx / r) * offset
Y_new = Y + (dy / r) * offset

	•	Displace pixels radially outward/inward.
	•	(dx/r, dy/r) = direction vector (unit vector from center).
	•	Multiply by offset to move pixels.

⸻

Remap Image

remapped = np.zeros_like(image_rgb)
for c in range(3):  # For each color channel
    channel = (image_rgb[:, :, c] * 255).astype(np.uint8)
    warped = cv2.remap(channel, X_new.astype(np.float32), Y_new.astype(np.float32),
                       interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    remapped[:, :, c] = warped.astype(np.float32) / 255.0

	•	For each color channel (R, G, B):
	•	Convert back to uint8 for OpenCV.
	•	Apply cv2.remap → moves pixels according to X_new, Y_new.
	•	BORDER_REFLECT mirrors at edges (no black borders).
	•	Convert back to [0,1] float.

⸻

Update Display

im.set_data(remapped)
return [im]

	•	Update the displayed image in the animation.

⸻

Step 6: Run Animation

ani = FuncAnimation(fig, update, frames=80, interval=100, blit=True)
plt.show()

	•	Animate for 80 frames.
	•	interval=100 → 100 ms between frames (≈10 fps).
	•	blit=True → efficient updating (only redraw changed parts).

⸻

🎨 What This Code Does
	•	Creates circular ripple waves spreading outward from the center.
	•	Each frame shows the ripple moving outward, just like dropping a stone into water.
	•	The wavefront fades at the edge, making it look realistic.

⸻

✅ In simple words:
This code animates ripples on an image by shifting pixels outward/inward in a circular sine wave pattern that grows with time, making the image look like it’s under water with waves spreading out.

⸻

👉 Do you want me to also modify this ripple animation to simulate multiple stones dropped (multiple ripple centers)?
"""