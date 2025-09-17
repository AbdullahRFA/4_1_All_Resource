import cv2
import numpy as np

# Load the sample image (grayscale)
image = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.02.tiff", cv2.IMREAD_GRAYSCALE)

# If lena.jpg isn't available, you can use any grayscale image or replace with your own path
if image is None:
    raise FileNotFoundError("Sample image not found. Replace with your own image path.")

# Compute the 2D Fourier Transform and shift the zero-frequency component to the center
dft = cv2.dft(np.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shifted = np.fft.fftshift(dft)

# Compute magnitude spectrum and apply logarithmic scaling
magnitude = cv2.magnitude(dft_shifted[:, :, 0], dft_shifted[:, :, 1])
log_magnitude = np.log1p(magnitude)

# Normalize for display
cv2.normalize(log_magnitude, log_magnitude, 0, 255, cv2.NORM_MINMAX)
log_magnitude = np.uint8(log_magnitude)

# Stack original and spectrum side by side
combined = np.hstack((image, log_magnitude))

# Display the result
cv2.imshow('Original Image and DFT Magnitude Spectrum', combined)
cv2.waitKey(0)
cv2.destroyAllWindows()
