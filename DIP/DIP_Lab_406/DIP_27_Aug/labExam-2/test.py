import cv2
import numpy as np
img = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.02.tiff", cv2.IMREAD_GRAYSCALE)

fourier = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
fourier_shift = np.fft.fftshift(fourier)

magnitude = cv2.magnitude(fourier_shift[:,:,0],fourier_shift[:,:,1])

magnitude = np.log1p(magnitude)

cv2.normalize(magnitude,magnitude, 0, 255, None, cv2.NORM_MINMAX)

cv2.imshow("DFT",magnitude)