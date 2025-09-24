import cv2

# Load Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # Draw bounding boxes
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Face Detection", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()
























"""
Perfect — this code is doing real-time face detection using OpenCV with your computer’s webcam. Let’s go step by step 👇

⸻

📌 Step-by-Step Explanation

1. Import OpenCV

import cv2

	•	Loads the OpenCV library, which is used for computer vision tasks like face detection, object tracking, etc.

⸻

2. Load the Haar Cascade Classifier

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

	•	Loads a pre-trained face detection model (haarcascade_frontalface_default.xml).
	•	Haar Cascade is a machine learning–based method trained on many face images.
	•	It can detect frontal faces in an image.

⸻

3. Start Webcam

cap = cv2.VideoCapture(0)

	•	Opens your default webcam (0).
	•	If you have multiple cameras, you can change 0 to 1, 2, etc.

⸻

4. Infinite Loop for Video Capture

while True:
    ret, frame = cap.read()
    if not ret:
        break

	•	Continuously captures video frames from the webcam.
	•	ret is a flag: True if a frame is successfully captured.
	•	frame is the actual image captured.
	•	If capturing fails, it breaks the loop.

⸻

5. Convert Frame to Grayscale

gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	•	Face detection works better in grayscale images.
	•	Converts the frame from BGR (color) to grayscale.

⸻

6. Detect Faces

faces = face_cascade.detectMultiScale(gray, 1.1, 5)

	•	Detects faces in the grayscale image.
	•	1.1 → scale factor: image is scaled down by 10% at each step to detect faces of different sizes.
	•	5 → minNeighbors: higher means fewer false detections but may miss some faces.
	•	Returns a list of rectangles: (x, y, w, h) for each detected face.

⸻

7. Draw Bounding Boxes

for (x, y, w, h) in faces:
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	•	Loops over all detected faces.
	•	Draws a green rectangle ((0, 255, 0)) around each face with thickness 2.

⸻

8. Show Result

cv2.imshow("Face Detection", frame)

	•	Displays the video stream with face bounding boxes in a window named “Face Detection”.

⸻

9. Quit on ‘q’ Key

if cv2.waitKey(1) == ord('q'):
    break

	•	Waits for key input.
	•	If you press q, it breaks the loop and stops capturing.

⸻

10. Release & Cleanup

cap.release()
cv2.waitKey(0)
cv2.destroyAllWindows()

	•	cap.release() → releases the webcam.
	•	cv2.waitKey(0) → waits for a key before closing all windows (but here it’s redundant since you already quit with ‘q’).
	•	cv2.destroyAllWindows() → closes all OpenCV windows.

⸻

🎯 In Simple Words

This program:
	1.	Opens your webcam.
	2.	Continuously captures frames.
	3.	Converts frames to grayscale.
	4.	Uses Haar Cascade to detect faces.
	5.	Draws green rectangles around them.
	6.	Displays the result live.
	7.	Stops when you press q.

⸻

⚡ Small tip: You don’t really need cv2.waitKey(0) at the end since you already quit with 'q'. Want me to simplify the code further by removing that unnecessary line?
"""