clear;close all;
img = imread('Lena512rgb.png');
% % noisy_image = imnoise(img,'salt & pepper',0.8);
% noisy_image = imnoise(img,'speckle',0.03);
% imshow(noisy_image,[]);

% img2 = imread('images_gauss/ImDenoised_2.png');
% PSNR = psnr(img2,img)
% 
% my_psnr = 10*log10(255^2/(sum((img(:)-img2(:)).^2)/512))
sigma = [2,5,10,20,30,40,60,80,100];
PSNR = zeros(size(sigma));
ind=1; 
for i=sigma
    img2 = imread(sprintf('images_rgb/ImNoisy_%d.png',i));
    [PSNR(ind),~] = psnr(img2,img);
    ind = ind + 1;
end