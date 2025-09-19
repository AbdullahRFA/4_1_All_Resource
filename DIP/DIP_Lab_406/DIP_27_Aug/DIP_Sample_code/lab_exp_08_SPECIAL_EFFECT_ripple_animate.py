import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_float
from skimage.transform import warp
from matplotlib.animation import FuncAnimation

# Load and normalize image
image = img_as_float(data.astronaut())  # Replace with your own image if desired
height, width, _ = image.shape

# Ripple parameters
base_amplitude = 5         # Ripple strength
frequency = 0.2            # Wave spacing
center_x, center_y = width // 2, height // 2  # Ripple origin

# Ripple distortion function
def ripple(coords, time):
    y, x = coords[:, 0], coords[:, 1]
    dx, dy = x - center_x, y - center_y
    r = np.sqrt(dx**2 + dy**2) + 1e-6  # Avoid division by zero

    # Expanding wavefront: only affects pixels within current radius
    wavefront_radius = time * 50  # Controls how far ripple has spread
    mask = r < wavefront_radius

    # Smooth transition near wavefront edge
    edge_fade = np.clip((wavefront_radius - r) / 20, 0, 1)

    # Ripple wave
    wave = np.sin(frequency * r - time)

    # Amplitude fades near edge
    amplitude = base_amplitude * edge_fade

    # Radial displacement
    offset = amplitude * wave
    x_new = x + (dx / r) * offset * mask
    y_new = y + (dy / r) * offset * mask
    return np.column_stack((y_new, x_new))

# Animation setup
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(image)
ax.axis('off')

def update(frame):
    time = frame * 0.3  # Controls ripple speed
    warped = warp(image, lambda coords: ripple(coords, time), mode='reflect')
    im.set_data(warped)
    return [im]

# Create animation
ani = FuncAnimation(fig, update, frames=80, interval=100, blit=True)
plt.show()