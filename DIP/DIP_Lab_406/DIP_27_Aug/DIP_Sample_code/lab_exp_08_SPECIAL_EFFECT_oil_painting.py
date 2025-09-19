import numpy as np
import matplotlib.pyplot as plt
from skimage import data, img_as_ubyte
from skimage.color import rgb2gray

def oil_paint_np(image, window_size=5, intensity_levels=20):
    image = img_as_ubyte(image)
    h, w, _ = image.shape
    pad = window_size // 2
    output = np.zeros_like(image)

    # Convert to grayscale and quantize intensity
    gray = rgb2gray(image)
    intensity = (gray * (intensity_levels - 1)).astype(np.uint8)

    # Pad image and intensity map
    padded_img = np.pad(image, ((pad, pad), (pad, pad), (0, 0)), mode='reflect')
    padded_intensity = np.pad(intensity, ((pad, pad), (pad, pad)), mode='reflect')

    for y in range(h):
        for x in range(w):
            hist = np.zeros((intensity_levels,), dtype=int)
            color_accum = np.zeros((intensity_levels, 3), dtype=int)

            for dy in range(-pad, pad + 1):
                for dx in range(-pad, pad + 1):
                    i = padded_intensity[y + pad + dy, x + pad + dx]
                    pixel = padded_img[y + pad + dy, x + pad + dx]
                    hist[i] += 1
                    color_accum[i] += pixel

            dominant = np.argmax(hist)
            output[y, x] = color_accum[dominant] // hist[dominant]

    return output

# Load image
image = data.astronaut()  # Full image, no cropping

# Apply oil painting effect
oil_painted = oil_paint_np(image, window_size=5, intensity_levels=20)

# Display both images with full scaling
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(image, aspect='auto')
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(oil_painted, aspect='auto')
axes[1].set_title("Oil Painting Effect")
axes[1].axis("off")

plt.tight_layout()
plt.show()