img = imread('nature.jpeg');  
img = rgb2gray(img);  
noisy_img = imnoise(img, 'salt & pepper', 0.05);  
denoised_img = medfilt2(noisy_img, [3 3]); 
figure;
subplot(2, 2, 1), imshow(img), title('Original Image');
subplot(2, 2, 2), imshow(noisy_img), title('Noisy (Salt & Pepper) Image');
subplot(2, 2, 4), imshow(denoised_img), title('Denoised (Median Filter) Image');