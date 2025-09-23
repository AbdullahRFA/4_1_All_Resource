import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Load image
image_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/group_pepple.avif"
original = cv2.imread(image_path)

# Run inference
results = model(original)

# YOLOv8 provides a built-in function to draw boxes and labels
annotated = results[0].plot()

# Convert BGR → RGB for matplotlib
original_rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Show original and detected images side by side
fig, axs = plt.subplots(2, 1, figsize=(16, 8))

axs[0].imshow(original_rgb)
axs[0].set_title("Original Image")
axs[0].axis("off")

axs[1].imshow(annotated_rgb)
axs[1].set_title("YOLOv8 Detection")
axs[1].axis("off")

plt.tight_layout()
plt.show()