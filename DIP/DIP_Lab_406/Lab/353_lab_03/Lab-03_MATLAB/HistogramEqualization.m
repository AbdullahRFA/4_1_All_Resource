% MATLAB code for histogram equalization

% Read the input image (grayscale)
input_image = imread('flower.jpeg'); % Replace 'your_image.jpg' with the actual image file

% Convert to grayscale if it is an RGB image
if size(input_image, 3) == 3
    input_image = rgb2gray(input_image);
end

% Display the original image
figure;
subplot(2, 2, 1);
imshow(input_image);
title('Input Image');

% Plot the histogram of the original image
subplot(2, 2, 2);
imhist(input_image);
title('Histogram of Input Image');

% Perform histogram equalization
equalized_image = histeq(input_image);

% Display the histogram-equalized image
subplot(2, 2, 3);
imshow(equalized_image);
title('Histogram-Equalized Image');

% Plot the histogram of the equalized image
subplot(2, 2, 4);
imhist(equalized_image);
title('Histogram of Equalized Image');