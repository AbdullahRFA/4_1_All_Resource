import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Load the first image from disk into a NumPy array
img = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.02.tiff")

# Print info about the first image
print("First image info: ")
print(type(img))       # Type of variable (usually numpy.ndarray)
print(img.shape)       # Shape of image array: (height, width, channels)
print(img.ndim)        # Number of dimensions (3 for color images)
print(img.size)        # Total number of elements (pixels * channels)
print(img.dtype)       # Data type of pixels (e.g., float32 or uint8)
print(img.nbytes)      # Total memory in bytes used by this image

# Load the second image
img2 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.03.tiff")

# Print info about the second image
print("Second image info: ")
print(type(img2))
print(img2.shape)
print(img2.ndim)
print(img2.size)
print(img2.dtype)
print(img2.nbytes)

# Access and print the RGB pixel values at location (row=10, col=10) in second image
print("Pixel values at position (10, 10) in second image:")
print("Red channel:", img2[10, 10, 0])
print("Green channel:", img2[10, 10, 1])
print("Blue channel:", img2[10, 10, 2])
