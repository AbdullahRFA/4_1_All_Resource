import cv2
import numpy as np
import pywt

# Load RGB image using OpenCV
image_bgr = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.02.tiff")  # Replace with your own image path
if image_bgr is None:
    raise FileNotFoundError("Image not found. Replace with a valid RGB image.")

# Convert to RGB and float32
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)

# Apply DWT to each channel
coeffs_r = pywt.dwt2(image_rgb[:, :, 0], 'db2')
coeffs_g = pywt.dwt2(image_rgb[:, :, 1], 'db2')
coeffs_b = pywt.dwt2(image_rgb[:, :, 2], 'db2')

# Unpack coefficients
cA_r, (cH_r, cV_r, cD_r) = coeffs_r
cA_g, (cH_g, cV_g, cD_g) = coeffs_g
cA_b, (cH_b, cV_b, cD_b) = coeffs_b

# Combine subbands into one image
def combine_subbands(cA, cH, cV, cD):
    rows, cols = cA.shape
    combined = np.zeros((rows * 2, cols * 2), dtype=np.float32)
    combined[0:rows, 0:cols] = cA
    combined[0:rows, cols:] = cH
    combined[rows:, 0:cols] = cV
    combined[rows:, cols:] = cD
    return combined

# Normalize to 0–255
def normalize(img):
    norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return np.uint8(norm)

# Combine and normalize each channel
combined_r = normalize(combine_subbands(cA_r, cH_r, cV_r, cD_r))
combined_g = normalize(combine_subbands(cA_g, cH_g, cV_g, cD_g))
combined_b = normalize(combine_subbands(cA_b, cH_b, cV_b, cD_b))

# Stack into RGB image
combined_rgb = cv2.merge([combined_r, combined_g, combined_b])

# Draw white borders around subbands
rows, cols = cA_r.shape
def draw_border_rgb(img, top, left, height, width, thickness=3, color=(255, 255, 255)):
    cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness)

draw_border_rgb(combined_rgb, 0, 0, rows, cols)                # cA
draw_border_rgb(combined_rgb, 0, cols, rows, cols)             # cH
draw_border_rgb(combined_rgb, rows, 0, rows, cols)             # cV
draw_border_rgb(combined_rgb, rows, cols, rows, cols)          # cD

# Resize original image to match combined layout
resized_rgb = cv2.resize(image_rgb, (cols * 2, rows * 2)).astype(np.uint8)

# Stack side-by-side
side_by_side = np.hstack((resized_rgb, combined_rgb))

# Display result
cv2.imshow('Original and Wavelet Subbands', side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()
