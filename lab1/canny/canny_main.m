clear
clc
close all
I = imread('lena512.bmp');  
% I = rgb2gray(I);
I = double(I);  
[height,width] = size(I);

%% 超参数定义
sigma = 1;%方差  
N = 5;%高斯核大小
%两个阈值
lowTh = 0.02;
highTh = 0.19;

%% 第一步：高斯滤波
% gausFilter = fspecial('gaussian', [5,5], sigma);  % matlab的高斯核
% J= imfilter(I, gausFilter, 'replicate');   % matlab的高斯滤波
% 使用多个sigma来做平滑
G = Gaussian_filter(I, N, sigma);

%% 第二步：求梯度
% sobel算子，效果比较一般
%[Gr, Grx, Gry] = sobel_dif(G);

% 直接一阶差分，效果不错
[Gr, Grx, Gry] = direct_dif(G);
  
%% 第三步：局部非极大值抑制
% 记住要沿着梯度方向比较，注意ppt里面3*3的核
K = NMS(Gr, Grx, Gry);

%% 第四步（1）：双阈值算法检测
[EdgeLarge,EdgeBetween] = biThreshold(K, highTh, lowTh);

%% 第四步（2）：把EdgeLarge的边缘连成连续的轮廓  
edge = Connect(EdgeLarge, EdgeBetween);

%% 结果可视化
figure,
subplot(221),imshow(G,[]),title('高斯滤波后的结果') 
subplot(222),imshow(Gr,[])  ,title('求梯度后的结果')  
subplot(223),imshow(K,[])  ,title('非极大值抑制后的结果')  
subplot(224),imshow(edge,[])  ,title('双阈值后的结果')
figure
% edgd(edge=1))=255;
%imwrite(edge,'./rep1/img/outkan.png')
% imwrite(G,'./rep1/img/1.png')
% imwrite(Gr,'./rep1/img/2.png')
% imwrite(K,'./rep1/img/3.png')
imshow(edge,[])


