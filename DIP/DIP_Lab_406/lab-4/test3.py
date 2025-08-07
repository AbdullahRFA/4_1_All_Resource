import numpy as np
import matplotlib.pyplot as plt
from skimage import io, color, exposure

# ---------- Load and Normalize RGB Image ----------
img_rgb = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0

# ---------- Convert RGB to HSV ----------
img_hsv = color.rgb2hsv(img_rgb)
H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]

# ---------- Equalize V Channel ----------
V_eq = exposure.equalize_hist(V)
hsv_eq = np.stack((H, S, V_eq), axis=2)
final_rgb = color.hsv2rgb(hsv_eq)

# ---------- Helper: Plot Histogram ----------
def plot_histogram(image, ax, title, colors=None):
    if image.ndim == 3:
        for i, color_name in enumerate(colors):
            hist, _ = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 1])
            ax.plot(hist, color=color_name, linewidth=1)
    else:
        hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])
        ax.plot(hist, color='black', linewidth=1)
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])

# ---------- Setup Subplot Grid ----------
fig, axes = plt.subplots(5, 2, figsize=(14, 18))

# --- 1. Original RGB ---
axes[0, 0].imshow(img_rgb)
axes[0, 0].set_title("Original RGB")
axes[0, 0].axis('off')
plot_histogram(img_rgb, axes[0, 1], "Histogram (RGB)", ['r', 'g', 'b'])

# --- 2. HSV Image ---
axes[1, 0].imshow(img_hsv)
axes[1, 0].set_title("HSV Image")
axes[1, 0].axis('off')
plot_histogram(img_hsv, axes[1, 1], "Histogram (HSV)", ['m', 'c', 'y'])

# --- 3. V Channel (Original) ---
axes[2, 0].imshow(V, cmap='gray')
axes[2, 0].set_title("V Channel (Original)")
axes[2, 0].axis('off')
plot_histogram(V, axes[2, 1], "Histogram (V Original)")

# --- 4. Equalized V ---
axes[3, 0].imshow(V_eq, cmap='gray')
axes[3, 0].set_title("Equalized V Channel")
axes[3, 0].axis('off')
plot_histogram(V_eq, axes[3, 1], "Histogram (V Equalized)")

# --- 5. Final RGB after HSV Equalization ---
axes[4, 0].imshow(final_rgb)
axes[4, 0].set_title("Final RGB (After HSV V Equalized)")
axes[4, 0].axis('off')
plot_histogram(final_rgb, axes[4, 1], "Histogram (Final RGB)", ['r', 'g', 'b'])

plt.tight_layout()
plt.show()

# Here is the complete Python code that:
# 	1.	Loads an RGB image.
# 	2.	Displays it with its RGB histogram.
# 	3.	Converts it to HSV, shows HSV image and histogram.
# 	4.	Separates H, S, V channels, shows V channel and its histogram.
# 	5.	Performs histogram equalization on the V channel.
# 	6.	Shows the equalized V channel with its histogram.
# 	7.	Merges the modified H, S, V’, shows the result and histogram.
# 	8.	Converts it back to RGB, and shows the final image with histogram.

# import numpy as np
# import matplotlib.pyplot as plt
# from skimage import io, color, exposure
#
# # 1. Load and Normalize RGB Image
# img_rgb = io.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")
# if img_rgb.dtype == np.uint8:
#     img_rgb = img_rgb / 255.0
#
# # ---------- Helper: Plot Histogram ----------
# def plot_histogram(image, ax, title, colors=None):
#     if image.ndim == 3:
#         for i, color_name in enumerate(colors):
#             hist, _ = np.histogram(image[:, :, i].flatten(), bins=256, range=[0, 1])
#             ax.plot(hist, color=color_name)
#     else:
#         hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 1])
#         ax.plot(hist, color='black')
#     ax.set_title(title)
#
# # 2. Show Original RGB Image & Histogram
# fig, axes = plt.subplots(2, 2, figsize=(12, 8))
# axes[0, 0].imshow(img_rgb)
# axes[0, 0].set_title("Original RGB Image")
# axes[0, 0].axis('off')
# plot_histogram(img_rgb, axes[0, 1], "RGB Histogram", ['r', 'g', 'b'])
#
# # 3. Convert RGB to HSV and Show
# img_hsv = color.rgb2hsv(img_rgb)
# axes[1, 0].imshow(img_hsv)
# axes[1, 0].set_title("HSV Image")
# axes[1, 0].axis('off')
# plot_histogram(img_hsv, axes[1, 1], "HSV Histogram", ['m', 'c', 'y'])
# plt.tight_layout()
# plt.show()
#
# # 4. Separate H, S, V channels
# H, S, V = img_hsv[:, :, 0], img_hsv[:, :, 1], img_hsv[:, :, 2]
#
# # 5. Show V channel and its histogram
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].imshow(V, cmap='gray')
# axes[0].set_title("V Channel (Original)")
# axes[0].axis('off')
# plot_histogram(V, axes[1], "V Channel Histogram")
# plt.tight_layout()
# plt.show()
#
# # 6. Equalize V channel
# V_eq = exposure.equalize_hist(V)
#
# # 7. Show Equalized V and Histogram
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].imshow(V_eq, cmap='gray')
# axes[0].set_title("Equalized V Channel")
# axes[0].axis('off')
# plot_histogram(V_eq, axes[1], "Equalized V Histogram")
# plt.tight_layout()
# plt.show()
#
# # 8. Merge Modified HSV (H, S, V')
# hsv_eq = np.stack((H, S, V_eq), axis=2)
#
# # 9. Show New HSV Image and Histogram
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].imshow(hsv_eq)
# axes[0].set_title("Modified HSV Image")
# axes[0].axis('off')
# plot_histogram(hsv_eq, axes[1], "Modified HSV Histogram", ['m', 'c', 'y'])
# plt.tight_layout()
# plt.show()
#
# # 10. Convert Back to RGB and Show with Histogram
# final_rgb = color.hsv2rgb(hsv_eq)
#
# fig, axes = plt.subplots(1, 2, figsize=(10, 4))
# axes[0].imshow(final_rgb)
# axes[0].set_title("Final RGB Image")
# axes[0].axis('off')
# plot_histogram(final_rgb, axes[1], "Final RGB Histogram", ['r', 'g', 'b'])
# plt.tight_layout()
# plt.show()
