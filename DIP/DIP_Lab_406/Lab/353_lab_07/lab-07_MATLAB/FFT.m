inputImage = imread('nature.jpeg');
if size(inputImage, 3) == 3
    inputImage = rgb2gray(inputImage);
end
fftImage = fft2(double(inputImage));
fftImageShifted = fftshift(fftImage);
magnitudeFFT = abs(fftImageShifted);
magnitudeFFTLog = log(1 + magnitudeFFT);
figure;
subplot(1, 2, 1);
imshow(inputImage);
title('Original Image');
subplot(1, 2, 2);
imshow(magnitudeFFTLog, []);
title('Magnitude Spectrum (FFT)');
