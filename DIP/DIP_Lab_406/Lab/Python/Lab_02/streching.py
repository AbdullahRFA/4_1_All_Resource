import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def contrast_stretching(image, m, E):
    normalized_image = np.array(image) / 255.0
    transformed_image = 1 / (1 + np.exp(-E * (normalized_image - m)))
    output_image = (transformed_image * 255).astype(np.uint8)
    return output_image

image_path = 'Image/nature.jpeg'   
image = Image.open(image_path)   

m = 0.5   
E = 10   

output_image = contrast_stretching(image, m, E)

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(image)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(output_image)
axes[1].set_title('Contrast Stretched Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()