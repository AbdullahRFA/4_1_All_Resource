% Read the input image
input_image = imread('flower.jpeg');

% Convert the image to grayscale if it is in RGB
if size(input_image, 3) == 3
    input_image = rgb2gray(input_image);
end

% Display the input image and its histogram
figure;
subplot(2, 3, 1);
imshow(input_image);
title('Input Image');

subplot(2, 3, 2);
imhist(input_image);
title('Histogram of Input Image');

% Calculate and plot the CDF of the input image
[counts, binLocations] = imhist(input_image);
cdf_input_image = cumsum(counts) / numel(input_image);

subplot(2, 3, 3);
plot(binLocations, cdf_input_image, 'LineWidth', 2);
title('CDF of Input Image');
xlabel('Pixel Intensity Values');
ylabel('CDF');

% Perform histogram equalization
equalized_image = histeq(input_image);

% Display the equalized image and its histogram
subplot(2, 3, 4);
imshow(equalized_image);
title('Histogram-Equalized Image');

% Plot the histogram of the equalized image
subplot(2, 3, 5);
imhist(equalized_image);
title('Histogram of Equalized Image');

% Calculate and plot the CDF of the equalized image
[counts_eq, binLocations_eq] = imhist(equalized_image);
cdf_equalized_image = cumsum(counts_eq) / numel(equalized_image);

subplot(2, 3, 6);
plot(binLocations_eq, cdf_equalized_image, 'LineWidth', 2);
title('CDF of Equalized Image');
xlabel('Pixel Intensity Values');
ylabel('CDF');