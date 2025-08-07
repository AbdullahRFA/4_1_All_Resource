import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# ------------------ Load the RGB image ------------------
img = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff")

# Normalize to [0,1] if image is in uint8 format
if img.dtype == np.uint8:
    img = img / 255.0

# ------------------ Show Original RGB and its Histogram ------------------
plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
plt.imshow(img)
plt.title("Original RGB Image")
plt.axis('off')

plt.subplot(1, 2, 2)
colors = ('r', 'g', 'b')
for i, color in enumerate(colors):
    hist, _ = np.histogram(img[:, :, i].flatten(), bins=256, range=[0, 1])
    plt.plot(hist, color=color)
plt.title("RGB Histograms")
plt.tight_layout()
plt.show()

# ------------------ Show R, G, B Grayscale Channels & Histograms ------------------
channels = ['Red', 'Green', 'Blue']
plt.figure(figsize=(15, 8))

for i in range(3):
    ch = img[:, :, i]

    plt.subplot(3, 2, 2*i + 1)
    plt.imshow(ch, cmap='gray')
    plt.title(f"{channels[i]} Channel (Grayscale)")
    plt.axis('off')

    plt.subplot(3, 2, 2*i + 2)
    hist, _ = np.histogram(ch.flatten(), bins=256, range=[0, 1])
    plt.plot(hist, color=colors[i])
    plt.title(f"{channels[i]} Histogram")

plt.tight_layout()
plt.show()

# ------------------ RGB to HSV Conversion ------------------
def rgb_to_hsv(img):
    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    max_val = np.max(img, axis=2)
    min_val = np.min(img, axis=2)
    delta = max_val - min_val + 1e-6

    H = np.zeros_like(max_val)
    S = delta / (max_val + 1e-6)
    V = max_val

    mask = (delta != 0)

    idx = (max_val == R) & mask
    H[idx] = ((G[idx] - B[idx]) / delta[idx]) % 6
    idx = (max_val == G) & mask
    H[idx] = ((B[idx] - R[idx]) / delta[idx]) + 2
    idx = (max_val == B) & mask
    H[idx] = ((R[idx] - G[idx]) / delta[idx]) + 4

    H = H / 6
    H[H < 0] += 1

    return np.stack((H, S, V), axis=2)

img_hsv = rgb_to_hsv(img)
H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

# ------------------ Histogram Equalization on V Channel ------------------
V_flat = V.flatten()
hist, bins = np.histogram(V_flat, bins=256, range=[0, 1])
cdf = hist.cumsum()
cdf_normalized = cdf / cdf[-1]
V_eq = np.interp(V_flat, bins[:-1], cdf_normalized).reshape(V.shape)

# ------------------ HSV to RGB Conversion ------------------
def hsv_to_rgb(hsv):
    H, S, V = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    H = H * 6
    C = V * S
    X = C * (1 - np.abs(H % 2 - 1))
    m = V - C

    R = np.zeros_like(H)
    G = np.zeros_like(H)
    B = np.zeros_like(H)

    cond1 = (H >= 0) & (H < 1)
    cond2 = (H >= 1) & (H < 2)
    cond3 = (H >= 2) & (H < 3)
    cond4 = (H >= 3) & (H < 4)
    cond5 = (H >= 4) & (H < 5)
    cond6 = (H >= 5) & (H <= 6)

    R[cond1], G[cond1], B[cond1] = C[cond1], X[cond1], 0
    R[cond2], G[cond2], B[cond2] = X[cond2], C[cond2], 0
    R[cond3], G[cond3], B[cond3] = 0, C[cond3], X[cond3]
    R[cond4], G[cond4], B[cond4] = 0, X[cond4], C[cond4]
    R[cond5], G[cond5], B[cond5] = X[cond5], 0, C[cond5]
    R[cond6], G[cond6], B[cond6] = C[cond6], 0, X[cond6]

    R += m
    G += m
    B += m

    return np.clip(np.stack((R, G, B), axis=2), 0, 1)

# Combine new V with original H and S
hsv_eq = np.stack((H, S, V_eq), axis=2)
img_rgb_eq = hsv_to_rgb(hsv_eq)

# ------------------ Display Results ------------------
plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(V, cmap='gray')
plt.title("Original V Channel")
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(V_eq, cmap='gray')
plt.title("Equalized V Channel")
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(img_rgb_eq)
plt.title("Enhanced RGB Image (V Equalized)")
plt.axis('off')

plt.subplot(2, 3, 4)
hist, _ = np.histogram(V.flatten(), bins=256, range=[0, 1])
plt.plot(hist, color='black')
plt.title("Histogram of Original V")

plt.subplot(2, 3, 5)
hist, _ = np.histogram(V_eq.flatten(), bins=256, range=[0, 1])
plt.plot(hist, color='black')
plt.title("Histogram of Equalized V")

plt.tight_layout()
plt.show()
