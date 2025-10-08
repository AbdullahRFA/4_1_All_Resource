import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float

# Load distorted image (replace with your own if needed)
image = img_as_float(io.imread('/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.01.tiff', as_gray=True))

# Parameters from the document (example values)
r = image.shape[1]  # width of image
y_center = image.shape[0] // 2  # vertical center

# Define trapezoid edge points (example values, adjust for your image)
x1, y1 = r - 1, y_center - 30
x2, y2 = 0, y_center - 70

# Compute a and b for trapezoid
a = y1 - ((y2 - y1) / (x2 - x1)) * x1
b = y1 + ((y2 - y1) / (x2 - x1)) * (r - x1)

# Stretch function: str(x) = ((b - a)/r) * x + a
def stretch(x):
    return ((b - a) / r) * x + a

# Apply vertical stretch to each column
corrected = np.zeros_like(image)
for x in range(r):
    for y in range(image.shape[0]):
        y_shifted = y - y_center
        y_new = int(y_shifted * stretch(x) / b + y_center)
        if 0 <= y_new < image.shape[0]:
            corrected[y, x] = image[y_new, x]

# Display original and corrected images
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image, cmap='gray')
axes[0].set_title("Original (Distorted)")
axes[0].axis('off')

axes[1].imshow(corrected, cmap='gray')
axes[1].set_title("Perspective Corrected")
axes[1].axis('off')

plt.tight_layout()
plt.show()
