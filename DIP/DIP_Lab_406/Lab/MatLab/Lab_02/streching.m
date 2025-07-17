% Load the image
image_path = 'nature.jpeg';
image = imread(image_path);


normalized_image = im2double(image);


m = 0.5;
E = 10;


transformed_image = 1 ./ (1 + exp(-E * (normalized_image - m)));


output_image = uint8(transformed_image * 255);


figure;
subplot(1, 2, 1);
imshow(image);
title('Original Image');

subplot(1, 2, 2);
imshow(output_image);
title('Contrast Stretched Image');

