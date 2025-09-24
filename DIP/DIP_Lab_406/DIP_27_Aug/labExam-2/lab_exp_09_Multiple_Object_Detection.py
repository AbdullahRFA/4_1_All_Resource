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

















"""
Great — this code is a YOLOv8 (You Only Look Once) object detection example simplified to use YOLO’s built-in plotting. Let’s break it down 👇

⸻

📌 Step-by-Step Explanation

1. Import Libraries

import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt

	•	cv2 (OpenCV): used for reading images and color conversions.
	•	YOLO (Ultralytics): state-of-the-art deep learning model for object detection.
	•	matplotlib.pyplot: used to display the image with bounding boxes.

⸻

2. Load YOLOv8 Model

model = YOLO("yolov8s.pt")

	•	Loads the YOLOv8-small pre-trained model (yolov8s.pt).
	•	If not already downloaded, it will fetch weights automatically.
	•	This model can detect 80+ object categories (from COCO dataset: people, cars, dogs, etc.).

⸻

3. Load the Input Image

image_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/group_pepple.avif"
image = cv2.imread(image_path)

	•	Reads the image from your computer.
	•	OpenCV loads images in BGR color format.

⸻

4. Run Inference (Object Detection)

results = model(image)

	•	Passes the image through the YOLOv8 model.
	•	results contains:
	•	Detected objects (bounding boxes).
	•	Class IDs (like “person”, “car”).
	•	Confidence scores.

⸻

5. Use YOLO’s Built-in Plotting

annotated = results[0].plot()

	•	YOLO automatically draws bounding boxes, labels, and confidence scores on the image.
	•	Returns the annotated image in BGR format (since it uses OpenCV internally).

⸻

6. Convert BGR → RGB

annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

	•	OpenCV images are in BGR, but Matplotlib expects RGB.
	•	Converts the image so that colors display correctly.

⸻

7. Display the Result

plt.figure(figsize=(10, 10))
plt.imshow(annotated_rgb)
plt.title("YOLOv8 Detection")
plt.axis("off")
plt.show()

	•	Shows the final annotated image (with boxes + labels).
	•	Title is set to "YOLOv8 Detection".
	•	axis("off") removes x/y axis ticks for a cleaner view.

⸻

🎯 In Simple Words

This program:
	1.	Loads a pre-trained YOLOv8 model.
	2.	Reads an input image.
	3.	Runs YOLO to detect objects in the image.
	4.	Automatically draws bounding boxes + labels using YOLO’s built-in function.
	5.	Displays the result nicely with Matplotlib.

⸻

✅ This version is simpler and shorter compared to manually drawing boxes — because YOLO’s .plot() handles all the annotations for you.

⸻

Do you want me to also show you how to extract detected objects (class name + confidence + bounding box coordinates) from results so you can use them later?
"""