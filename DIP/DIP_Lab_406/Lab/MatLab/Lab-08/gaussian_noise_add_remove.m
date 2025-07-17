img = imread('nature.jpeg'); 
img = rgb2gray(img);  
noisy_img = imnoise(img, 'gaussian', 0, 0.01);  
arithmetic_mean_img = imfilter(noisy_img, fspecial('average', [3 3]));
[rows, cols] = size(noisy_img);
geo_img = double(noisy_img);
for i = 2:rows-1
    for j = 2:cols-1
        patch = geo_img(i-1:i+1, j-1:j+1);  
        geo_mean = exp(mean(mean(log(double(patch) + 1)))); 
        geo_img(i,j) = geo_mean - 1;  
    end
end
geo_img = uint8(geo_img);
figure;
subplot(2, 2, 1), imshow(img), title('Original Image');
subplot(2, 2, 2), imshow(noisy_img), title('Noisy (Gaussian) Image');
subplot(2, 2, 3), imshow(arithmetic_mean_img), title('Arithmetic Mean Filtered Image');
subplot(2, 2, 4), imshow(geo_img), title('Geometric Mean Filtered Image');