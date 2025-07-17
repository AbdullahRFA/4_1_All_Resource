import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# Load the image
img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.05.tiff")

# Separate the RGB channels
r = img1[:, :, 0]
g = img1[:, :, 1]
b = img1[:, :, 2]

# Set spacing between subplots
plt.subplots_adjust(hspace=0.5, wspace=0.5)

# Plot the original image
plt.subplot(2, 2, 1)
plt.title('Original Image')
plt.imshow(img1)
plt.axis('off')

# Red channel histogram
hist_r, bins_r = np.histogram(r.ravel(), bins=256, range=(0, 256))
plt.subplot(2, 2, 2)
plt.title('Red Histogram')
plt.bar(bins_r[:-1], hist_r, color='red')

# Green channel histogram
hist_g, bins_g = np.histogram(g.ravel(), bins=256, range=(0, 256))
plt.subplot(2, 2, 3)
plt.title('Green Histogram')
plt.bar(bins_g[:-1], hist_g, color='green')

# Blue channel histogram
hist_b, bins_b = np.histogram(b.ravel(), bins=256, range=(0, 256))
plt.subplot(2, 2, 4)
plt.title('Blue Histogram')
plt.bar(bins_b[:-1], hist_b, color='blue')

# Display all plots
plt.show()
