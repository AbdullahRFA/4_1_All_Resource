function [H, S, I] = rgb2hsi_components(rgb_image)
    [rows, cols, ~] = size(rgb_image);
    H = zeros(rows, cols);
    S = zeros(rows, cols);
    I = zeros(rows, cols);
    rgb_image = im2double(rgb_image);
    R = rgb_image(:, :, 1);
    G = rgb_image(:, :, 2);
    B = rgb_image(:, :, 3);
    I = (R + G + B) / 3;
    min_RGB = min(min(R, G), B);
    S = 1 - (3 ./ (R + G + B + eps)) .* min_RGB;
    theta = acos((0.5 * ((R - G) + (R - B))) ./ sqrt((R - G).^2 + (R - B).*(G - B) + eps));
    H(B > G) = 2 * pi - theta(B > G);
    H(G >= B) = theta(G >= B);
    H = H / (2 * pi);  
end
rgb_image = imread('nature.jpeg');  
[H, S, I] = rgb2hsi_components(rgb_image);   

figure;
subplot(2, 2, 1), imshow(rgb_image), title('RGB Image');
subplot(2, 2, 2), imshow(H), title('Hue');
subplot(2, 2, 3), imshow(S), title('Saturation');
subplot(2, 2, 4), imshow(I), title('Intensity');
