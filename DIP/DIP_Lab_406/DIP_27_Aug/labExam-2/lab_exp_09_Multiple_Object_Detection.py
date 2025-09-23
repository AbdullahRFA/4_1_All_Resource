import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Load YOLOv8 model
model = YOLO("yolov8s.pt")

# Load image
image_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/group_pepple.avif"
image = cv2.imread(image_path)

# Run inference
results = model(image)

# Use YOLO's built-in plotting (auto draws bounding boxes + labels)
annotated = results[0].plot()

# Convert BGR → RGB for matplotlib
annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

# Show result
plt.figure(figsize=(10, 10))
plt.imshow(annotated_rgb)
plt.title("YOLOv8 Detection")
plt.axis("off")
plt.show()