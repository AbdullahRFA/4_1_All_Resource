inputImage = imread('nature.jpeg');
if size(inputImage, 3) == 3
    inputImage = rgb2gray(inputImage);
end
inputImage = im2double(inputImage);
logImage = log(1 + inputImage);
fftImage = fft2(logImage);
fftImageShifted = fftshift(fftImage);
[M, N] = size(inputImage);
D0 = 30; 
n = 2; 
[X, Y] = meshgrid(1:N, 1:M);
centerX = ceil(N/2);
centerY = ceil(M/2);
D = sqrt((X - centerX).^2 + (Y - centerY).^2);
H = 1 - exp(-(D.^2 / (2 * D0^2))); 
filteredFFT = fftImageShifted .* H;
filteredFFTShiftedBack = ifftshift(filteredFFT);
inverseFFT = ifft2(filteredFFTShiftedBack);
filteredImage = real(inverseFFT);
finalImage = exp(filteredImage) - 1;
figure;
subplot(1, 2, 1);
imshow(inputImage, []);
title('Original Image');
subplot(1, 2, 2);
imshow(finalImage, []);
title('Homomorphic Filtered Image');
