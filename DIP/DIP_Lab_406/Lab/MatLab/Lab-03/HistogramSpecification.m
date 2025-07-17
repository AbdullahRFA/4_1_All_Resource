input_image = imread('nature.jpeg');
reference_image = imread('flower.jpeg'); 
if size(input_image, 3) == 3
    input_image = rgb2gray(input_image);
end
if size(reference_image, 3) == 3
    reference_image = rgb2gray(reference_image);
end
matched_image = imhistmatch(input_image, reference_image);


figure;
subplot(2, 3, 1);
imshow(input_image);
title('Input Image');

subplot(2, 3, 2);
imshow(reference_image);
title('Reference Image');

subplot(2, 3, 3);
imshow(matched_image);
title('Histogram-Matched(I)');

subplot(2, 3, 4);
imhist(input_image);
title('Input Image');

subplot(2, 3, 5);
imhist(reference_image);
title('Reference Image');

subplot(2, 3, 6);
imhist(matched_image);
title('Matched Image');