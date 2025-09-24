import cv2
import matplotlib.pyplot as plt

# Load image
image_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff"
image = cv2.imread(image_path)

# Convert to grayscale (needed for face detection)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Load Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Detect faces
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

# Draw bounding boxes on original image
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green box

# Convert BGR → RGB for matplotlib
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Show result
plt.imshow(image_rgb)
plt.title(f"Detected Faces: {len(faces)}")
plt.axis('off')
plt.show()













"""
Perfect — let’s break down your code step by step so it’s crystal clear 👇

⸻

📌 Step-by-Step Explanation

1. Import Libraries

import cv2
import matplotlib.pyplot as plt

	•	cv2 (OpenCV): used for image handling and face detection.
	•	matplotlib.pyplot: used to display the image with bounding boxes.

⸻

2. Load Image

image_path = "/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff"
image = cv2.imread(image_path)

	•	Reads the image from your system into a NumPy array.
	•	By default, OpenCV loads images in BGR (Blue, Green, Red) format.

⸻

3. Convert to Grayscale

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

	•	Face detection works on intensity information (not colors).
	•	Converts the color image into grayscale for processing.

⸻

4. Load Haar Cascade (Face Detector)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	•	Loads a pre-trained Haar Cascade model for detecting faces.
	•	This XML file contains patterns/features trained to detect human faces.

⸻

5. Detect Faces

faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

	•	detectMultiScale → detects objects (here, faces).
	•	Parameters:
	•	gray → input grayscale image.
	•	scaleFactor=1.1 → how much the image size is reduced at each scale (helps detect faces at different distances).
	•	minNeighbors=5 → how many neighbors a candidate rectangle should have to be retained (higher = stricter).
	•	Output: a list of bounding boxes → (x, y, w, h) for each detected face.

⸻

6. Draw Bounding Boxes

for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

	•	Loops through each detected face.
	•	Draws a green rectangle around the face on the original image.
	•	(x, y) → top-left corner of the rectangle.
	•	(x + w, y + h) → bottom-right corner.
	•	(0, 255, 0) → green color in BGR.
	•	2 → line thickness.

⸻

7. Convert BGR → RGB

image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

	•	OpenCV uses BGR, but Matplotlib expects RGB.
	•	This fixes the color order before displaying.

⸻

8. Display the Result

plt.imshow(image_rgb)
plt.title(f"Detected Faces: {len(faces)}")
plt.axis('off')
plt.show()

	•	Displays the final image using Matplotlib.
	•	Shows the number of detected faces in the title.
	•	Hides axes for a clean view.

⸻

🎯 In Simple Words

This program:
	1.	Loads an image.
	2.	Converts it to grayscale.
	3.	Uses a Haar Cascade face detector to find faces.
	4.	Draws green rectangles around detected faces.
	5.	Displays the result with Matplotlib.

⸻

✅ If you run this with an image containing multiple people, you’ll see all their faces highlighted with green boxes.

⸻

Do you want me to also show you how to print the coordinates (x, y, w, h) of each detected face alongside the image?
"""