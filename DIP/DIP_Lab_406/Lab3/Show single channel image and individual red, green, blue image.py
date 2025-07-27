import matplotlib.pyplot as plt       # Import matplotlib for plotting images
import matplotlib.image as mpimg      # Import matplotlib's image module to read images
import numpy as np                    # Import NumPy for numerical operations (not used explicitly here but good practice)
from matplotlib.pyplot import title

img1 = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.06.tiff")

r = img1[:, :, 0]

g = img1[:, :, 1]

b = img1[:, :, 2]

red = img1.copy()
green = img1.copy()
blue = img1.copy()
red[:, :, [1,2]] = 0
green[:, :, [0,2]] = 0
blue[:, :, [0,1]] = 0

output_single_channel = [img1, red, green, blue]
titles_for_single_channel = ['Original Image', 'Red image', 'Green image', 'Blue image']

output_gray_scale = [img1, r, g, b]

titles = ['Original Image', 'Red Channel', 'Green Channel', 'Blue Channel']

for i in range(4):
    plt.subplot(2, 2, i + 1)
    plt.axis('off')
    plt.title(titles[i])

    if i == 0:
        plt.imshow(output_gray_scale[i])
    else:
        plt.imshow(output_gray_scale[i], cmap='gray')


plt.tight_layout()
plt.show()

for i in range(4):
    plt.subplot(2,2,i+1)
    plt.axis('off')
    plt.title(titles_for_single_channel[i])

    if i == 0:
        plt.imshow(output_single_channel[i])
    else:
        plt.imshow(output_single_channel[i])

plt.tight_layout()
plt.show()
