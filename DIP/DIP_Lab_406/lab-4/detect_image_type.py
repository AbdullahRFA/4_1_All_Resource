from PIL import Image
import numpy as np
import os

def detect_image_type(image_path):
    try:
        img = Image.open(image_path)
        mode = img.mode
        ext = os.path.splitext(image_path)[1].lower()
        unique_vals = np.unique(np.array(img))

        print(f"\nImage: {os.path.basename(image_path)}")
        print(f"Mode: {mode}")
        print(f"Extension: {ext}")
        print(f"Image Size: {img.size}")

        if mode == '1' or (len(unique_vals) == 2 and set(unique_vals) <= {0, 255}):
            return "Binary Image"
        elif mode == 'L':
            if ext == '.pgm':
                return "PGM (Grayscale Image)"
            return "Grayscale Image"
        elif mode == 'RGB':
            if ext == '.ppm':
                return "PPM (RGB Color Image)"
            return "RGB Image"
        elif mode == 'P':
            return "Indexed Color Image (Palette)"
        else:
            return f"Unknown Image Type (Mode: {mode})"
    except Exception as e:
        return f"Error reading image: {e}"

# ------------- Example Usage -------------

# Change to your image path
image_paths = [
    "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff",  # RGB
    "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/5.3.01.tiff",  # Grayscale
    "converted_rgb.ppm",  # PPM
    "converted_gray.pgm"  # PGM
]

for path in image_paths:
    result = detect_image_type(path)
    print(f"Detected Type: {result}")