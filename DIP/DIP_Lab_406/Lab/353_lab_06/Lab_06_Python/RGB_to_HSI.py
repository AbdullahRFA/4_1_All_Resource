import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
def rgb2hsi_components(rgb_image):
    rgb_image = np.array(rgb_image, dtype=np.float64) / 255.0
    R = rgb_image[:, :, 0]
    G = rgb_image[:, :, 1]
    B = rgb_image[:, :, 2]
    I = (R + G + B) / 3
    min_RGB = np.minimum(np.minimum(R, G), B)
    S = 1 - (3 / (R + G + B + 1e-8)) * min_RGB
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G) ** 2 + (R - B) * (G - B)) + 1e-8
    theta = np.arccos(num / den)
    H = np.copy(theta)
    H[B > G] = 2 * np.pi - H[B > G]
    H = H / (2 * np.pi)   
    return H, S, I
rgb_image = Image.open('nature.jpeg')   
H, S, I = rgb2hsi_components(rgb_image)
fig, ax = plt.subplots(2, 2, figsize=(10, 8))
ax[0, 0].imshow(rgb_image)
ax[0, 0].set_title('RGB Image')
ax[0, 0].axis('off')
ax[0, 1].imshow(H, cmap='hsv')
ax[0, 1].set_title('Hue')
ax[0, 1].axis('off')
ax[1, 0].imshow(S, cmap='gray')
ax[1, 0].set_title('Saturation')
ax[1, 0].axis('off')
ax[1, 1].imshow(I, cmap='gray')
ax[1, 1].set_title('Intensity')
ax[1, 1].axis('off')
plt.tight_layout()
plt.show()
