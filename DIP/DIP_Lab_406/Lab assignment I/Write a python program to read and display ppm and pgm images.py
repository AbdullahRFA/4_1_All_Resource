from PIL import Image
import matplotlib.pyplot as plt
import os

# ---------- Step 1: Load RGB Image ----------
input_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff"  # Change this to your RGB image path
img_rgb = Image.open(input_path).convert('RGB')

# ---------- Step 2: Save as PPM ----------
ppm_path = "converted_image.ppm"
img_rgb.save(ppm_path, format='PPM')

# ---------- Step 3: Convert to Grayscale and Save as PGM ----------
img_gray = img_rgb.convert('L')  # 'L' mode is grayscale
pgm_path = "converted_image.pgm"
img_gray.save(pgm_path, format='PPM')  # Pillow uses 'PPM' for PGM/PPM/PNM

# ---------- Step 4: Display All Images ----------
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Original RGB
axes[0].imshow(img_rgb)
axes[0].set_title("Original RGB Image")
axes[0].axis('off')

# PPM
ppm_img = Image.open(ppm_path)
axes[1].imshow(ppm_img)
axes[1].set_title("PPM Image (Color)")
axes[1].axis('off')

# PGM
pgm_img = Image.open(pgm_path)
axes[2].imshow(pgm_img, cmap='gray')
axes[2].set_title("PGM Image (Grayscale)")
axes[2].axis('off')

plt.tight_layout()
plt.show()