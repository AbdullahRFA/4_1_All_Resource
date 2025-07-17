inputImage = imread('nature.jpeg');
if size(inputImage, 3) == 3
    inputImage = rgb2gray(inputImage);
end
se = strel('square', 3);
dilatedImage = imdilate(inputImage, se);
figure;
subplot(1, 2, 1);
imshow(inputImage);
title('Original Image');
subplot(1, 2, 2);
imshow(dilatedImage);
title('Dilated Image');
