import matplotlib.pyplot as plt       # Import matplotlib for plotting images
import matplotlib.image as mpimg      # Import matplotlib's image module to read images
import numpy as np                    # Import NumPy for numerical operations (not used explicitly here but good practice)

# Load image from the specified path into a NumPy array
img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.06.tiff")

# Extract the Red color channel by slicing all rows and columns, but only the 0th channel
r = img1[:, :, 0]

# Extract the Green color channel by slicing all rows and columns, but only the 1st channel
g = img1[:, :, 1]

# Extract the Blue color channel by slicing all rows and columns, but only the 2nd channel
b = img1[:, :, 2]

# Create a list to hold the original image and the three color channels separately
output = [img1, r, g, b]

# Create a list of titles to label the subplots accordingly
titles = ['Original Image', 'Red Channel', 'Green Channel', 'Blue Channel']

# Loop over each item (0 to 3) to plot original and individual color channels
for i in range(4):
    plt.subplot(2, 2, i + 1)      # Create a 2x2 grid of subplots, and select subplot (i+1)
    plt.axis('off')               # Hide axis ticks and labels for a cleaner image display
    plt.title(titles[i])          # Set the title for the subplot

    if i == 0:
        plt.imshow(output[i])     # Display the original image in color
    else:
        plt.imshow(output[i], cmap='gray')  # Display single color channel images in grayscale for clarity

plt.tight_layout()  # Adjust subplot spacing to prevent overlap of titles and images
plt.show()          # Render and display the figure window with all subplots
