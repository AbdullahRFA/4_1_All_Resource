image = imread('nature.jpeg');
if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end
skipFactor = 4;   
skippedImage = grayImage(1:skipFactor:end, 1:skipFactor:end);
figure;
subplot(1, 2, 1);
imshow(grayImage);
title('Original Image');
subplot(1, 2, 2);
imshow(skippedImage);
title('Pixel Skipped Image');
