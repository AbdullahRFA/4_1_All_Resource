import os
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, util, restoration
from skimage.draw import polygon
from skimage.color import gray2rgb
import cv2
import logging

# --- Logging setup for debugging ---
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Main Code ---

# Load a sample grayscale image (camera photo)
original_gray = data.camera()
logging.debug("Loaded camera image")

# Convert grayscale to RGB (optional, for visualization)
image_rgb = gray2rgb(original_gray)
logging.debug("Converted image to RGB")

# Initialize OpenCV Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces in the image
faces = face_cascade.detectMultiScale(original_gray, scaleFactor=1.1, minNeighbors=5)
logging.info(f"Detected {len(faces)} faces")

# Draw bounding boxes around detected faces
bounded_gray = original_gray.copy()
for (x, y, w, h) in faces:
    cv2.rectangle(bounded_gray, (x, y), (x + w, y + h), 255, 2)
logging.debug("Drew bounding boxes")

# Add Gaussian noise
noisy_gray = util.random_noise(original_gray, mode='gaussian', var=0.02)
noisy_gray = (255 * noisy_gray).astype(np.uint8)
logging.debug("Added Gaussian noise")

# Copy noisy image for selective denoising
denoised_gray = noisy_gray.copy()

# Apply denoising only to detected face regions
if len(faces) > 0:
    logging.debug("Applying Non-Local Means denoising to face regions")
    filtered = restoration.denoise_nl_means(
        noisy_gray,
        h=0.15 * np.std(noisy_gray),
        fast_mode=True,
        patch_size=5,
        patch_distance=6,
        channel_axis=None
    )
    filtered = (255 * filtered).astype(np.uint8)

    # Replace only the face regions with denoised patches
    for (x, y, w, h) in faces:
        rr, cc = polygon([y, y, y + h, y + h], [x, x + w, x + w, x], shape=original_gray.shape)
        mask = np.zeros(original_gray.shape, dtype=bool)
        mask[rr, cc] = True
        denoised_gray[mask] = filtered[mask]
    logging.debug("Completed selective denoising")

# --- Display Results ---
fig, axes = plt.subplots(1, 4, figsize=(20, 5))
titles = ['Original Grayscale', 'Face-Bounded Grayscale', 'Noisy Image', 'Face-Denoised Image']
images = [original_gray, bounded_gray, noisy_gray, denoised_gray]

for ax, img, title in zip(axes, images, titles):
    ax.imshow(img, cmap='gray')
    ax.set_title(title)
    ax.axis('off')

plt.tight_layout()
plt.show()
logging.debug("Displayed all results successfully")

# --- Cleanup ---
del faces, bounded_gray, noisy_gray, denoised_gray
if 'filtered' in locals():
    del filtered
if 'mask' in locals():
    del mask
logging.debug("Cleanup complete")