import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, img_as_float, img_as_ubyte
from sklearn.cluster import KMeans
from scipy.ndimage import convolve, gaussian_filter

# -------- Load Grayscale and RGB Images --------
gray_img = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/5.3.01.tiff", as_gray=True)
rgb_img = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")

# Normalize
gray_img = img_as_float(gray_img)
rgb_img = img_as_float(rgb_img)

# -------- Convert RGB to Indexed Image (via KMeans) --------
pixels = rgb_img.reshape(-1, 3)
kmeans = KMeans(n_clusters=16, random_state=42)
labels = kmeans.fit_predict(pixels)
palette = kmeans.cluster_centers_

indexed_img = palette[labels].reshape(rgb_img.shape)

# --------- Define Filters ---------
avg_kernel = np.ones((3, 3)) / 9

high_pass_kernel = np.array([
    [-1, -1, -1],
    [-1,  8, -1],
    [-1, -1, -1]
])

# --------- Filtering Function ---------
def apply_filters(image):
    result = {}
    result['average'] = convolve(image, avg_kernel)
    result['high_pass'] = np.clip(convolve(image, high_pass_kernel), 0, 1)
    result['gaussian'] = gaussian_filter(image, sigma=1)
    return result

# --------- Apply Filtering ---------
gray_filtered = apply_filters(gray_img)

rgb_filtered = {
    key: np.stack([apply_filters(rgb_img[:, :, i])[key] for i in range(3)], axis=2)
    for key in ['average', 'high_pass', 'gaussian']
}

indexed_filtered = {
    key: np.stack([apply_filters(indexed_img[:, :, i])[key] for i in range(3)], axis=2)
    for key in ['average', 'high_pass', 'gaussian']
}

# --------- Display Results ---------
titles = ['Original', 'Average', 'High-Pass', 'Gaussian']
images = [
    (gray_img, gray_filtered),
    (rgb_img, rgb_filtered),
    (indexed_img, indexed_filtered)
]
labels = ['Grayscale', 'RGB', 'Indexed']

plt.figure(figsize=(16, 12))

for i, (original, filtered_dict) in enumerate(images):
    for j, key in enumerate(['original', 'average', 'high_pass', 'gaussian']):
        plt_idx = i * 4 + j + 1
        plt.subplot(3, 4, plt_idx)

        if key == 'original':
            img_to_show = original
        else:
            img_to_show = filtered_dict[key]

        plt.imshow(img_to_show, cmap='gray' if i == 0 else None)
        plt.title(f"{labels[i]} - {titles[j]}")
        plt.axis('off')

plt.tight_layout()
plt.show()