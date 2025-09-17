import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from scipy.signal import convolve2d

# Step 1: Load grayscale image using OpenCV
image_path = '/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.08.tiff'


image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
if image is None:
    raise FileNotFoundError("Image not found. Replace with your own grayscale image.")
image = image.astype(np.float32) / 255.0  # Normalize to [0, 1]

# Step 2: Create motion blur kernel
def motion_blur_kernel(size, angle):
    kernel = np.zeros((size, size), dtype=np.float32)
    kernel[size // 2, :] = np.ones(size)
    rot_mat = cv2.getRotationMatrix2D((size // 2, size // 2), angle, 1)
    kernel = cv2.warpAffine(kernel, rot_mat, (size, size))
    kernel /= np.sum(kernel)
    return kernel

kernel_size = 15
angle = 30
psf = motion_blur_kernel(kernel_size, angle)

# Step 3: Apply motion blur using convolution
blurred = convolve2d(image, psf, mode='same', boundary='wrap')

# Step 4: Add Gaussian noise
noise = np.random.normal(0, np.sqrt(0.001), image.shape)
noisy_blurred = np.clip(blurred + noise, 0, 1)

# Step 5a: Pseudo-Inverse Filter
def pseudo_inverse_filter(img, psf, threshold=0.01):
    img_fft = fft2(img)
    psf_fft = fft2(psf, s=img.shape)
    psf_fft[np.abs(psf_fft) < threshold] = threshold
    result_fft = img_fft / psf_fft
    result = np.real(ifft2(result_fft))
    return np.clip(result, 0, 1)

# Step 5b: Wiener Filter
def wiener_filter(img, psf, K):
    img_fft = fft2(img)
    psf_fft = fft2(psf, s=img.shape)
    psf_fft_conj = np.conj(psf_fft)
    wiener_fft = psf_fft_conj / (np.abs(psf_fft)**2 + K)
    result_fft = wiener_fft * img_fft
    result = np.real(ifft2(result_fft))
    return np.clip(result, 0, 1)

# Step 6: Restore images
restored_inverse = pseudo_inverse_filter(noisy_blurred, psf)
restored_wiener = wiener_filter(noisy_blurred, psf, K=0.01)

# Step 7: Display results
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
titles = ['Original', 'Motion Blurred', 'Blurred + Noise', 'Inverse Filter', 'Wiener Filter']
images = [image, blurred, noisy_blurred, restored_inverse, restored_wiener]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
