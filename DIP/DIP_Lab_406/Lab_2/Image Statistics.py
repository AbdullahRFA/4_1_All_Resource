import matplotlib.pyplot as plt       # Import matplotlib for plotting (not used here, but commonly imported)
import matplotlib.image as mpimg      # Import matplotlib image module to read images
import numpy as np                    # Import NumPy for numerical operations on arrays

# Load image from the specified path into a NumPy array
img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.04.tiff")

# Print the minimum pixel value in the image
print("Min:", img1.min())

# Print the maximum pixel value in the image
print("Max:", img1.max())

# Print the mean (average) pixel value in the image
print("Mean:", img1.mean())

# Print the median pixel value using NumPy's median function
print("Median:", np.median(img1))

# Print the average pixel value using NumPy's average function (similar to mean)
print("Average:", np.average(img1))

# Print the mean pixel value again using NumPy's mean function explicitly
print("Mean (np):", np.mean(img1))

# Print the standard deviation of the pixel values (measure of spread)
print("Standard Deviation:", np.std(img1))

# Print the variance of the pixel values (square of standard deviation)
print("Variance:", np.var(img1))
