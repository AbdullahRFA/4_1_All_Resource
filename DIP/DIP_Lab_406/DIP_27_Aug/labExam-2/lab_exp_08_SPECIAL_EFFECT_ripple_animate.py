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
