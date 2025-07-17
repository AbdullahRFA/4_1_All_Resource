image = imread('nature.jpeg');
if size(image, 3) == 3
    grayImage = rgb2gray(image);
else
    grayImage = image;
end
filterSize = 3;   
filteredImage = medfilt2(grayImage, [filterSize filterSize]);
figure;
subplot(1, 2, 1);
imshow(grayImage);
title('Original Image');
subplot(1, 2, 2);
imshow(filteredImage);
title('Median Filtered Image');
