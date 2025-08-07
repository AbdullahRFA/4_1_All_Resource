import cv2
import numpy as np

# -------- Step 1: Read the image in RGB --------
img_rgb = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.2.07.tiff")
# -------- Step 2: Convert RGB to BGR --------
img_bgr_converted = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

# -------- Step 3: Create VGR image --------
hsv_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)  # Convert RGB to HSV
V = hsv_img[:, :, 2]  # Extract Value channel
R, G, _ = cv2.split(img_rgb)  # Use original RGB channels (ignore original B)
VGR = cv2.merge([V, G, R])  # Replace B with V

# -------- Step 5: Combine and Show All --------
combined = np.hstack((img_rgb, img_bgr_converted, VGR))
cv2.imshow("RGB | BGR | VGR", combined)
cv2.waitKey(0)
cv2.destroyAllWindows()