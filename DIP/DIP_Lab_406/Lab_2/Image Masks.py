import matplotlib.pyplot as plt       # For displaying images
import matplotlib.image as mpimg      # For reading images
import numpy as np                    # For numerical operations

# Load the image
img1 = plt.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.05.tiff")

# Make a writable copy of the image
img1 = img1.copy()

# Get image dimensions
nrows, ncols, channels = img1.shape

# Create row and column indices
row, col = np.ogrid[:nrows, :ncols]

# Compute center of the image
cnt_row, cnt_col = nrows / 2, ncols / 2

# Create an outer disk mask: True for pixels outside the center circle
outer_disk_mask = ((row - cnt_row) ** 2 + (col - cnt_col) ** 2 > (nrows / 2) ** 2)

# Apply the mask: set pixels outside the center circle to 0 (black)
img1[outer_disk_mask] = 0

# Display the masked image
plt.imshow(img1)
plt.axis('off')
plt.title('Masked Image')
plt.show()