img = imread('nature.jpeg');  
img = rgb2gray(img);  
F = fft2(double(img));
F_shifted = fftshift(F);
magnitude_spectrum = log(1 + abs(F_shifted));
figure;
subplot(2, 2, 1), imshow(img, []), title('Original Image');
subplot(2, 2, 2), imshow(magnitude_spectrum, []), title('Magnitude Spectrum');
function F_filtered = notch_filter(F_shifted, center, radius)
    [rows, cols] = size(F_shifted);
    mask = ones(rows, cols); 
    [X, Y] = meshgrid(1:cols, 1:rows);
    
    mask(((X - center(2)).^2 + (Y - center(1)).^2) < radius^2) = 0;
    
    F_filtered = F_shifted .* mask;
end
F_filtered = notch_filter(F_shifted, [130, 130], 15); 
F_filtered = notch_filter(F_filtered, [170, 170], 15);  
F_ishifted = ifftshift(F_filtered);
img_filtered = ifft2(F_ishifted);
img_filtered = abs(img_filtered);
subplot(2, 2, 3), imshow(img_filtered, []), title('Filtered Image');
