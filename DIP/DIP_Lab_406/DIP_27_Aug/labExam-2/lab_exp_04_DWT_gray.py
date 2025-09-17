import cv2
import numpy as np
import pywt

# Load grayscale image using OpenCV
camera_img = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.02.tiff", cv2.IMREAD_GRAYSCALE)
if camera_img is None:
    raise FileNotFoundError("Image not found. Replace with your own grayscale image path.")

# Resize to 512x512 for consistency (optional)
camera_img = cv2.resize(camera_img, (512, 512))

# Apply DWT using PyWavelets
coeffs2 = pywt.dwt2(camera_img, 'db2')
cA, (cH, cV, cD) = coeffs2

# Combine subbands into one image (2x2 block)
rows, cols = cA.shape
combined = np.zeros((rows * 2, cols * 2), dtype=np.float32)

combined[0:rows, 0:cols] = cA       # Top-left
combined[0:rows, cols:] = cH        # Top-right
combined[rows:, 0:cols] = cV        # Bottom-left
combined[rows:, cols:] = cD         # Bottom-right

# Normalize combined image to 0–255 for display
combined_norm = cv2.normalize(combined, None, 0, 255, cv2.NORM_MINMAX)
combined_norm = np.uint8(combined_norm)

# Function to draw borders
def draw_border(img, top, left, height, width, thickness=3, color=255):
    cv2.rectangle(img, (left, top), (left + width, top + height), color, thickness)

# Draw borders around subbands
draw_border(combined_norm, 0, 0, rows, cols)                # cA
draw_border(combined_norm, 0, cols, rows, cols)             # cH
draw_border(combined_norm, rows, 0, rows, cols)             # cV
draw_border(combined_norm, rows, cols, rows, cols)          # cD

# Resize original image to match combined layout
camera_resized = cv2.resize(camera_img, (cols * 2, rows * 2))

# Stack side-by-side
side_by_side = np.hstack((camera_resized, combined_norm))

# Display result
cv2.imshow('Original Image and Wavelet Subbands', side_by_side)
cv2.waitKey(0)
cv2.destroyAllWindows()
