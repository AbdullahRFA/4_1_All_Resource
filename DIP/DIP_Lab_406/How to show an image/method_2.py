import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

# Load the image
img = mpimg.imread('/Users/abdullahnazmus-sakib/Desktop/995F9C93-2B51-4DF9-A785-FB27757BCA5D.png')

# Print the full matrix
np.set_printoptions(threshold=np.inf)  # Show entire array
print(img)

# plt.imshow(img)
# plt.title("Class Routine")
# plt.axis('off')
# plt.show()