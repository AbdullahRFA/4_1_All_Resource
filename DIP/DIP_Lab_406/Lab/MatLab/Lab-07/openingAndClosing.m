inputImage = imread('nature.jpeg');
if size(inputImage, 3) == 3
    inputImage = rgb2gray(inputImage);
end
se = strel('square', 3);
openedImage = imopen(inputImage, se);
closedImage = imclose(inputImage, se);
figure;
subplot(1, 3, 1);
imshow(inputImage);
title('Original Image');
subplot(1, 3, 2);
imshow(openedImage);
title('Opened Image');
subplot(1, 3, 3);
imshow(closedImage);
title('Closed Image');
