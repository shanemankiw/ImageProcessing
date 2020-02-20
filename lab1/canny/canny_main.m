clear
clc
close all
I = imread('lena512.bmp');  
% I = rgb2gray(I);
I = double(I);  
[height,width] = size(I);

%% ����������
sigma = 1;%����  
N = 5;%��˹�˴�С
%������ֵ
lowTh = 0.02;
highTh = 0.19;

%% ��һ������˹�˲�
% gausFilter = fspecial('gaussian', [5,5], sigma);  % matlab�ĸ�˹��
% J= imfilter(I, gausFilter, 'replicate');   % matlab�ĸ�˹�˲�
% ʹ�ö��sigma����ƽ��
G = Gaussian_filter(I, N, sigma);

%% �ڶ��������ݶ�
% sobel���ӣ�Ч���Ƚ�һ��
%[Gr, Grx, Gry] = sobel_dif(G);

% ֱ��һ�ײ�֣�Ч������
[Gr, Grx, Gry] = direct_dif(G);
  
%% ���������ֲ��Ǽ���ֵ����
% ��סҪ�����ݶȷ���Ƚϣ�ע��ppt����3*3�ĺ�
K = NMS(Gr, Grx, Gry);

%% ���Ĳ���1����˫��ֵ�㷨���
[EdgeLarge,EdgeBetween] = biThreshold(K, highTh, lowTh);

%% ���Ĳ���2������EdgeLarge�ı�Ե��������������  
edge = Connect(EdgeLarge, EdgeBetween);

%% ������ӻ�
figure,
subplot(221),imshow(G,[]),title('��˹�˲���Ľ��') 
subplot(222),imshow(Gr,[])  ,title('���ݶȺ�Ľ��')  
subplot(223),imshow(K,[])  ,title('�Ǽ���ֵ���ƺ�Ľ��')  
subplot(224),imshow(edge,[])  ,title('˫��ֵ��Ľ��')
figure
% edgd(edge=1))=255;
%imwrite(edge,'./rep1/img/outkan.png')
% imwrite(G,'./rep1/img/1.png')
% imwrite(Gr,'./rep1/img/2.png')
% imwrite(K,'./rep1/img/3.png')
imshow(edge,[])


