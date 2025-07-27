# Importing necessary libraries
import matplotlib.pyplot as plt               # For plotting images and histograms
import matplotlib.image as mpimg              # For reading image files
import numpy as np                            # For numerical operations
from skimage import exposure, img_as_ubyte    # For histogram equalization and image conversion

# Load the RGB image
img_rgb = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")

# Normalize image to [0, 1] range if it's in uint8 format
if img_rgb.dtype == np.uint8:
    img_rgb = img_rgb / 255.0

# Split the RGB image into individual Red, Green, and Blue channels
R = img_rgb[:, :, 0]
G = img_rgb[:, :, 1]
B = img_rgb[:, :, 2]

# HSI to RGB Conversion Function
def hsi_to_rgb(H, S, I):
    # Convert H (Hue) from normalized [0,1] to radians [0, 2π]
    H = H * 2 * np.pi

    # Initialize R, G, B arrays with zeros having the same shape as H
    R, G, B = np.zeros(H.shape), np.zeros(H.shape), np.zeros(H.shape)

    # Sector 1: 0 <= H < 2π/3 (Red-Green region)
    sector1 = (H >= 0) & (H < 2*np.pi/3)
    h1 = H[sector1]  # Get Hue values in this sector
    B[sector1] = I[sector1] * (1 - S[sector1])  # Compute Blue component
    R[sector1] = I[sector1] * (1 + S[sector1] * np.cos(h1) / np.cos(np.pi/3 - h1))  # Compute Red component
    G[sector1] = 3 * I[sector1] - (R[sector1] + B[sector1])  # Compute Green using the identity: I = (R+G+B)/3

    # Sector 2: 2π/3 <= H < 4π/3 (Green-Blue region)
    sector2 = (H >= 2*np.pi/3) & (H < 4*np.pi/3)
    h2 = H[sector2] - 2*np.pi/3  # Normalize Hue to start from 0 in this sector
    R[sector2] = I[sector2] * (1 - S[sector2])  # Compute Red component
    G[sector2] = I[sector2] * (1 + S[sector2] * np.cos(h2) / np.cos(np.pi/3 - h2))  # Compute Green component
    B[sector2] = 3 * I[sector2] - (R[sector2] + G[sector2])  # Compute Blue

    # Sector 3: 4π/3 <= H <= 2π (Blue-Red region)
    sector3 = (H >= 4*np.pi/3) & (H <= 2*np.pi)
    h3 = H[sector3] - 4*np.pi/3  # Normalize Hue to start from 0 in this sector
    G[sector3] = I[sector3] * (1 - S[sector3])  # Compute Green component
    B[sector3] = I[sector3] * (1 + S[sector3] * np.cos(h3) / np.cos(np.pi/3 - h3))  # Compute Blue component
    R[sector3] = 3 * I[sector3] - (G[sector3] + B[sector3])  # Compute Red

    # Stack R, G, B into a single 3D array and clip values to [0,1] to ensure valid image data
    return np.clip(np.stack((R, G, B), axis=2), 0, 1)


# Define a function to convert RGB to HSI color space
def rgb_to_hsi(R, G, B):
    epsilon = 1e-6  # Small value to prevent division by zero

    # Intensity component: average of R, G, B
    I = (R + G + B) / 3.0

    # Minimum of R, G, B for each pixel
    min_rgb = np.minimum(np.minimum(R, G), B)

    # Saturation component calculation
    S = 1 - (3 / (R + G + B + epsilon)) * min_rgb

    # Numerator and denominator for Hue calculation
    num = 0.5 * ((R - G) + (R - B))
    den = np.sqrt((R - G)**2 + (R - B)*(G - B)) + epsilon

    # Angle for Hue in radians, clipped to [-1, 1] to avoid numerical errors
    theta = np.arccos(np.clip(num / den, -1, 1))

    # Convert radians to degrees
    H = np.degrees(theta)

    # Adjust Hue for pixels where B > G
    H[B > G] = 360 - H[B > G]

    # Normalize Hue to range [0, 1]
    H = H / 360

    return H, S, I

# Convert RGB image to HSI color space
H, S, I = rgb_to_hsi(R, G, B)

# Stack H, S, and I components to form a pseudo-color HSI image
hsi_image = np.stack((H, S, I), axis=2)

# Convert Intensity channel to 8-bit unsigned integer for histogram processing
I_uint8 = img_as_ubyte(I)

# Compute histogram of original intensity values
hist_I, _ = np.histogram(I_uint8.flatten(), bins=256, range=[0, 256])

# Apply histogram equalization to the Intensity component
I_eq = exposure.equalize_hist(I_uint8)

# Convert equalized intensity to uint8 format for display
I_eq_uint8 = img_as_ubyte(I_eq)

# Compute histogram of equalized intensity values
hist_eq, _ = np.histogram(I_eq_uint8.flatten(), bins=256, range=[0, 256])


# Replace original I with equalized I
hsi_eq = (H, S, I_eq)

# Convert enhanced HSI back to RGB
enhanced_rgb = hsi_to_rgb(H, S, I_eq)


# ----------- Plotting ---------------

# Set figure size
plt.figure(figsize=(18, 14))

# Show the original RGB image
plt.subplot(3, 3, 1)
plt.imshow(img_rgb)
plt.title("Original RGB Image")
plt.axis('off')

# Show pseudo-color HSI image
plt.subplot(3, 3, 2)
plt.imshow(hsi_image)
plt.title("HSI Image (Pseudo-color)")
plt.axis('off')

# Show the original Intensity channel
plt.subplot(3, 3, 3)
plt.imshow(I_uint8, cmap='gray')
plt.title("Original Intensity (I)")
plt.axis('off')

# Show the Hue channel in HSV colormap
plt.subplot(3, 3, 4)
plt.imshow(H, cmap='hsv')
plt.title("Hue Channel")
plt.axis('off')

# Show the Saturation channel in grayscale
plt.subplot(3, 3, 5)
plt.imshow(S, cmap='gray')
plt.title("Saturation Channel")
plt.axis('off')

# Show the equalized Intensity channel
plt.subplot(3, 3, 6)
plt.imshow(I_eq_uint8, cmap='gray')
plt.title("Equalized Intensity (I)")
plt.axis('off')

# Show the histogram of the original Intensity channel
plt.subplot(3, 3, 7)
plt.plot(hist_I, color='black')
plt.title("Histogram of Original I")

# Show the histogram of the equalized Intensity channel
plt.subplot(3, 3, 8)
plt.plot(hist_eq, color='black')
plt.title("Histogram of Equalized I")

# Enhanced RGB image
plt.subplot(3, 3, 9)
plt.imshow(enhanced_rgb)
plt.title("Enhanced RGB Image (After I Histogram Equalization)")
plt.axis('off')


# Adjust layout to prevent overlap
plt.tight_layout()

# Display all plots
plt.show()
