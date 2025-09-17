import cv2
import numpy as np
import matplotlib.pyplot as plt

def oil_paint_cv(image, window_size=5, intensity_levels=20):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB
    image = cv2.normalize(image.astype(np.float32), None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    h, w, _ = image.shape
    pad = window_size // 2
    output = np.zeros_like(image)

    # Convert to grayscale and quantize intensity
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    intensity = (gray * (intensity_levels - 1) / 255).astype(np.uint8)

    # Pad image and intensity map
    padded_img = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT)
    padded_intensity = cv2.copyMakeBorder(intensity, pad, pad, pad, pad, cv2.BORDER_REFLECT)

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

# Load image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Apply oil painting effect
oil_painted = oil_paint_cv(image_bgr, window_size=5, intensity_levels=20)

# Display results
fig, axes = plt.subplots(1, 2, figsize=(14, 7))
axes[0].imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
axes[0].set_title("Original Image")
axes[0].axis("off")

axes[1].imshow(oil_painted)
axes[1].set_title("Oil Painting Effect")
axes[1].axis("off")

plt.tight_layout()
plt.show()
