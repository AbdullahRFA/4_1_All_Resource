

# 🖼️ 1. Pillow (PIL Fork) – Most common and simple
```python
from PIL import Image

# Open an image file
image = Image.open("your_image.jpg")

# Show the image
image.show()
```


	•	✅ Lightweight and built-in image viewer
	•	🛠️ You can also manipulate (resize, rotate, crop, etc.)

## To install:

``` 
pip install Pillow
```


⸻

# 🖥️ 2. OpenCV – Powerful for computer vision
```python
import cv2

# Read the image
image = cv2.imread("your_image.jpg")

# Display the image in a window
cv2.imshow("Image", image)
cv2.waitKey(0)  # Wait for any key to close
cv2.destroyAllWindows()
```


## To install:

```
pip install opencv-python
```


⸻

# 🐍 3. Matplotlib – Good for displaying inline in notebooks


```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load and display the image
img = mpimg.imread("your_image.jpg")
plt.imshow(img)
plt.axis('off')  # Hide axes
plt.show()
```

## To install:

```
pip install matplotlib
```


⸻
