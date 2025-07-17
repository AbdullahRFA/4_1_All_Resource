import matplotlib.pyplot as plt
import matplotlib.image as mpimg

img = mpimg.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/4.1.01.tiff")
plt.title("Image")
plt.imshow(img)
plt.axis('off')
plt.show()
