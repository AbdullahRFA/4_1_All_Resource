function rgb_components_to_grayscale(image)
    
    image = double(image) / 255;
    R = image(:, :, 1);
    G = image(:, :, 2);
    B = image(:, :, 3);
    grayscale_image = 0.2989 * R + 0.5870 * G + 0.1140 * B;
    figure;
    
    subplot(3, 2, 2), imshow(R), title('Red Component');
    subplot(3, 2, 3), imshow(G), title('Green Component');
    subplot(3, 2, 4), imshow(B), title('Blue Component');
    subplot(3, 2, 5), imshow(grayscale_image), title('Grayscale Image');
    subplot(3, 2, 1), imshow(image), title('Original Image');
end
image = imread('nature.jpeg');
rgb_components_to_grayscale(image);