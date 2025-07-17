image = imread('flower.jpeg');
gray = rgb2gray(image);
[centers, radii, metric] = imfindcircles(gray, [20 80], 'Sensitivity', 0.9, 'EdgeThreshold', 0.1);
imshow(image);
hold on;
viscircles(centers, radii, 'EdgeColor', 'r');
title('Detected Circles using Hough Transform');
hold off;
