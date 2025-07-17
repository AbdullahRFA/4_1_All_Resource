image = imread('nature.jpeg');
if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end
newSize = [300 400];   
resizedImage = imresize(grayImage, newSize, 'nearest');
figure;
subplot(1, 2, 1);
imshow(grayImage);
title('Original Image');
subplot(1, 2, 2);
imshow(resizedImage);
title('Resized Image (Replication Method)');
