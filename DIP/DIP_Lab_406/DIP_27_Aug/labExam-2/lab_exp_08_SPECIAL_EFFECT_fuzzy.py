import cv2
import numpy as np
import matplotlib.pyplot as plt

# Step 1: Load RGB image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'

image_bgr = cv2.imread(image_path)  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert BGR to RGB and normalize to [0, 1]
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
height, width,_= image_rgb.shape

# Step 2: Define neighborhood size
window_size = 7
pad = window_size // 2

# Step 3: Pad image to handle borders
padded_image = cv2.copyMakeBorder(image_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)

# Step 4: Create output image
fuzzy_image = np.zeros_like(image_rgb)

# Step 5: Apply random neighborhood replacement
for y in range(height):
    for x in range(width):
        dy = np.random.randint(-pad, pad + 1)
        dx = np.random.randint(-pad, pad + 1)
        fuzzy_image[y, x] = padded_image[y + pad + dy, x + pad + dx]

# Step 6: Display results
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(fuzzy_image)
axes[1].set_title('Fuzzy Image (Random Neighborhood)')
axes[1].axis('off')

plt.tight_layout()
plt.show()













"""
Absolutely, Abdullah! This code simulates a **"fuzzy effect"** on an image by randomly replacing each pixel with a neighboring pixel — creating a soft, randomized blur that mimics uncertainty or imprecision. Here's a step-by-step breakdown with conceptual clarity:

---

## 🧩 Step-by-Step Explanation

### 🔹 Step 1: Load and Normalize the Image
```python
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
```
- Loads the image using OpenCV (which defaults to BGR).
- Converts it to RGB and normalizes pixel values to the range [0, 1] for better processing.

---

### 🔹 Step 2: Define Neighborhood Size
```python
window_size = 7
pad = window_size // 2
```
- Sets a **7×7 neighborhood** around each pixel.
- `pad = 3` means each pixel will consider ±3 pixels in both directions.

---

### 🔹 Step 3: Pad the Image
```python
padded_image = cv2.copyMakeBorder(image_rgb, pad, pad, pad, pad, cv2.BORDER_REFLECT)
```
- Pads the image using **reflection** to avoid border issues.
- Ensures that even edge pixels have a full neighborhood to sample from.

---

### 🔹 Step 4: Create Output Image
```python
fuzzy_image = np.zeros_like(image_rgb)
```
- Initializes an empty image with the same shape as the original.

---

### 🔹 Step 5: Apply Random Neighborhood Replacement
```python
for y in range(height):
    for x in range(width):
        dy = np.random.randint(-pad, pad + 1)
        dx = np.random.randint(-pad, pad + 1)
        fuzzy_image[y, x] = padded_image[y + pad + dy, x + pad + dx]
```
- For each pixel `(y, x)`:
  - Randomly selects a neighboring pixel within the 7×7 window.
  - Replaces the current pixel with that neighbor.
- This introduces **random local variation**, creating a fuzzy or smeared look.

---

### 🔹 Step 6: Display Results
```python
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(image_rgb)
axes[1].imshow(fuzzy_image)
```
- Shows the original and fuzzy image side by side using `matplotlib`.

---

## 🎨 What Does This "Fuzzy Effect" Do?

- It **destroys sharp edges** and **blurs textures** by replacing pixels with random neighbors.
- Unlike Gaussian blur (which is weighted and smooth), this is **non-deterministic** and **chaotic**.
- It mimics **uncertainty** — useful for simulating noise, anonymization, or artistic distortion.

---

Would you like to extend this with a fuzzy logic-based edge detector or segmentation next? I can help you build that with `skfuzzy` or custom rules.

"""