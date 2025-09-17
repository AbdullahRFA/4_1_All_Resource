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
