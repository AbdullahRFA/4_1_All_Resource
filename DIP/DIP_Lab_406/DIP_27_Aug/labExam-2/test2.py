import cv2
import numpy as np

# ------------------- Read Image -------------------
img = cv2.imread("/Users/abdullahnazmus-sakib/Desktop/4_1_All_Resource/DIP/DIP_Lab_406/images/7.1.02.tiff", cv2.IMREAD_GRAYSCALE)

# img = cv2.imread('img.png', cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original", img)

# ------------------- Histogram Equalization -------------------
he = cv2.equalizeHist(img)
cv2.imshow("Histogram Equalized", he)

# ------------------- CLAHE -------------------
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
clahe_img = clahe.apply(img)
cv2.imshow("CLAHE", clahe_img)

# ------------------- Gamma using LUT -------------------
gamma = 0.6
invGamma = 1.0 / gamma
lut = np.array([((i/255.0)**invGamma)*255 for i in range(256)]).astype('uint8')
gamma_img = cv2.LUT(img, lut)
cv2.imshow("Gamma Corrected (0.6)", gamma_img)

# ------------------- Gaussian Blur (low-pass) -------------------
gauss = cv2.GaussianBlur(img, (5,5), 1.0)
cv2.imshow("Gaussian Blur", gauss)

# ------------------- Laplacian (high-pass) -------------------
lap = cv2.Laplacian(img, cv2.CV_64F)
lap = np.uint8(np.clip(np.abs(lap), 0, 255))
cv2.imshow("Laplacian (High-Pass)", lap)

# ------------------- Unsharp Mask (Sharpening) -------------------
blur = cv2.GaussianBlur(img, (9,9), 10.0)
unsharp = cv2.addWeighted(img, 1.5, blur, -0.5, 0)
cv2.imshow("Unsharp Mask (Sharpened)", unsharp)

# ------------------- Otsu Threshold -------------------
ret, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imshow("Otsu Threshold", otsu)

# ------------------- Canny Edges -------------------
edges = cv2.Canny(img, 100, 200)
cv2.imshow("Canny Edges", edges)


# ------------------- Fourier Filtering (Gaussian High-Pass) -------------------
f = np.fft.fft2(img.astype(float))
fshift = np.fft.fftshift(f)
rows, cols = img.shape
crow, ccol = rows//2, cols//2

# Create Gaussian high-pass filter
D0 = 30.0
u = np.arange(-crow, rows-crow)[:,None]
v = np.arange(-ccol, cols-ccol)[None,:]
D2 = u**2 + v**2
H = 1 - np.exp(-D2/(2*(D0**2)))

# Apply filter
G = H * fshift
g_ishift = np.fft.ifftshift(G)
img_back = np.fft.ifft2(g_ishift)
img_back = np.real(img_back)
img_back = np.uint8(np.clip(img_back, 0, 255))
cv2.imshow("Fourier Gaussian High-Pass", img_back)


# ------------------- Homomorphic Filtering -------------------
imgf = img.astype(np.float32) + 1.0
log_img = np.log(imgf)

F = np.fft.fft2(log_img)
Fshift = np.fft.fftshift(F)

gammaL, gammaH = 0.5, 2.0
D0 = 30.0
H2 = (gammaH - gammaL) * (1 - np.exp(-D2 / (2 * (D0**2)))) + gammaL

G2 = H2 * Fshift

g_ishift2 = np.fft.ifftshift(G2)
img_back2 = np.fft.ifft2(g_ishift2)
img_back2 = np.exp(np.real(img_back2)) - 1.0
img_back2 = np.uint8(np.clip(img_back2, 0, 255))
cv2.imshow("Homomorphic Filter Output", img_back2)


# ------------------- Wait and Close -------------------
cv2.waitKey(0)
cv2.destroyAllWindows()