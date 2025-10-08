import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, uniform_filter
from skimage import io, img_as_float
from skimage.filters.rank import mean
from skimage.morphology import square

# Load grayscale image
image = img_as_float(io.imread('/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.06.tiff', as_gray=True))

# 1. Geometric Mean Filter
def geometric_mean_filter(image, size=3):
    def geo_mean(values):
        return np.prod(values)**(1.0/len(values))
    return generic_filter(image, geo_mean, size=(size, size))

# 2. Alpha-Trimmed Mean Filter
def alpha_trimmed_mean_filter(image, size=3, d=2):
    def alpha_trimmed(values):
        sorted_vals = np.sort(values)
        trimmed = sorted_vals[d//2 : -d//2]
        return np.mean(trimmed)
    return generic_filter(image, alpha_trimmed, size=(size, size))

# 3. Root-Mean-Square (RMS) Filter
def rms_filter(image, size=3):
    def rms(values):
        return np.sqrt(np.mean(np.square(values)))
    return generic_filter(image, rms, size=(size, size))

# 4. Unsharp Masking
def unsharp_mask(image, size=3, amount=1.0):
    blurred = uniform_filter(image, size=size)
    mask = image - blurred
    sharpened = image + amount * mask
    return np.clip(sharpened, 0, 1)

# 5. Kuwahara Filter (Simplified)
def kuwahara_filter(image, size=5):
    image_ubyte = (image * 255).astype(np.uint8)
    return mean(image_ubyte, square(size)) / 255.0

# Apply all filters
filters = {
    "Original": image,
    "Geometric Mean": geometric_mean_filter(image),
    "Alpha-Trimmed Mean": alpha_trimmed_mean_filter(image),
    "RMS Filter": rms_filter(image),
    "Unsharp Mask": unsharp_mask(image),
    "Kuwahara Filter": kuwahara_filter(image)
}

# Display results
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (title, result) in zip(axes.flatten(), filters.items()):
    ax.imshow(result, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()





























"""
Absolutely! Here's a detailed line-by-line explanation of the complete Python script that implements five filtering and enhancement algorithms:

---

## 🧠 Import Libraries

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import generic_filter, uniform_filter
from skimage import io, img_as_float
from skimage.filters.rank import mean
from skimage.morphology import square
```

- `numpy`: For numerical operations and array handling.
- `matplotlib.pyplot`: For displaying images and plots.
- `scipy.ndimage.generic_filter`: Applies custom filters over an image.
- `scipy.ndimage.uniform_filter`: Performs simple averaging (used in unsharp masking).
- `skimage.io`: Loads images from files or URLs.
- `skimage.img_as_float`: Converts image to float format (range 0–1).
- `skimage.filters.rank.mean`: Applies rank-based mean filter (used in Kuwahara).
- `skimage.morphology.square`: Creates a square-shaped structuring element for filtering.

---

## 📷 Load Grayscale Image

```python
image = img_as_float(io.imread('https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png', as_gray=True))
```

- Loads the Lenna image from the web.
- Converts it to grayscale and float format for processing.

---

## 🧮 Geometric Mean Filter

```python
def geometric_mean_filter(image, size=3):
    def geo_mean(values):
        return np.prod(values)**(1.0/len(values))
    return generic_filter(image, geo_mean, size=(size, size))
```

- Computes the geometric mean of pixel values in a neighborhood.
- `np.prod(values)`: Multiplies all values.
- `**(1.0/len(values))`: Takes the nth root (geometric mean).
- `generic_filter`: Applies this function over the image.

---

## ✂️ Alpha-Trimmed Mean Filter

```python
def alpha_trimmed_mean_filter(image, size=3, d=2):
    def alpha_trimmed(values):
        sorted_vals = np.sort(values)
        trimmed = sorted_vals[d//2 : -d//2]
        return np.mean(trimmed)
    return generic_filter(image, alpha_trimmed, size=(size, size))
```

- Sorts pixel values in a neighborhood.
- Trims `d` values (half from each end).
- Averages the remaining values.

---

## 📈 Root-Mean-Square (RMS) Filter

```python
def rms_filter(image, size=3):
    def rms(values):
        return np.sqrt(np.mean(np.square(values)))
    return generic_filter(image, rms, size=(size, size))
```

- Squares each pixel value.
- Averages the squares.
- Takes the square root of the result.

---

## 🔍 Unsharp Masking

```python
def unsharp_mask(image, size=3, amount=1.0):
    blurred = uniform_filter(image, size=size)
    mask = image - blurred
    sharpened = image + amount * mask
    return np.clip(sharpened, 0, 1)
```

- Blurs the image using a uniform filter.
- Subtracts the blurred image from the original to get a "mask".
- Adds the mask back to the original to sharpen.
- `np.clip`: Ensures pixel values stay between 0 and 1.

---

## 🧠 Kuwahara Filter (Simplified)

```python
def kuwahara_filter(image, size=5):
    image_ubyte = (image * 255).astype(np.uint8)
    return mean(image_ubyte, square(size)) / 255.0
```

- Converts image to 8-bit format.
- Applies a mean filter using a square window.
- Converts result back to float format.

---

## 🖼️ Apply All Filters

```python
filters = {
    "Original": image,
    "Geometric Mean": geometric_mean_filter(image),
    "Alpha-Trimmed Mean": alpha_trimmed_mean_filter(image),
    "RMS Filter": rms_filter(image),
    "Unsharp Mask": unsharp_mask(image),
    "Kuwahara Filter": kuwahara_filter(image)
}
```

- Dictionary to store results of each filter for easy display.

---

## 🎨 Display Results

```python
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
for ax, (title, result) in zip(axes.flatten(), filters.items()):
    ax.imshow(result, cmap='gray')
    ax.set_title(title)
    ax.axis('off')
plt.tight_layout()
plt.show()
```

- Creates a 2×3 grid of subplots.
- Displays each filtered image with its title.
- `cmap='gray'`: Ensures grayscale rendering.
- `plt.tight_layout()`: Prevents overlap.
- `plt.show()`: Renders the plot.

---

Let me know if you want to save the results, apply them to your own image, or explore more filters like median or Gaussian!
"""