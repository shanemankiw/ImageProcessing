clc; 
close all; 
clear all; 

img_gray = imread('Lena512.png');

%img_gray = imresize(img_gray,[256,256]);
figure(1); 
imshow(img_gray); 
title('original'); 


% 加噪声
sigma = 10; 
img_noise = double(img_gray)+sigma * randn(size(img_gray));

figure(2); 
imshow(img_noise / 255, []); 
title('noise_added'); 

noisy_image = imnoise(img_gray,'gaussian', 0, sigma*sigma/255/255);
figure(3);
imshow(img_noise, []); 
title('imnoise_image'); 

img_denoise = BM3D_Gray(img_noise, 2, sigma, 1); 

figure; 
imshow(img_denoise / 255, []); 
title('second_stage_result'); 

function img_denoise = BM3D_Gray(img_noise, tran_mode, sigma, isDisplay)
% 参考文献：An Analysis and Implementation of the BM3D Image Denoising Method
% Inputs:
%        img_noise: 灰度噪声图像，必须为矩形方阵
%        tran_mode: = 0, fft; = 1, dct; = 2, dwt, = 3, db1
% Outputs:
%        img_denoise: 去噪图像
%
if ~exist('tran_mode', 'var')
    tran_mode = 0;
end

if ~exist('sigma', 'var')
    sigma = 10;
end

if ~exist('isDisplay', 'var')
    isDisplay = 0;
end

[row,col] = size(img_noise);

% First Step 参数
kHard           = 8;          % 块大小
pHard           = 3;          % 块移动间隔
lambda_distHard = 0;          % 求相似的距离时，变换后，收缩的阈值
nHard           = 16;         % 搜索窗口大小
NHard           = 28;         % 最多相似块个数
tauHard         = 2500;       % 最大的相似距离for fft

% kaiser窗口的参数，实际上并没有特别大的影响
beta=2;
Wwin2D = kaiser(kHard, beta) * kaiser(kHard, beta)';

% Second Step参数
kWien           = kHard;
pWien           = pHard;
lambda_distWien = lambda_distHard;
nWien           = nHard;
NWien           = NHard;
tauWien         = tauHard;
sigma2          = sigma*sigma;

if(tran_mode==0)        %fft
    lambda2d=400;
    lambda1d=500;
    lambda2d_wie=50;
    lambda1d_wie=500;
elseif(tran_mode == 1)  %dct
    lambda2d=50;
    lambda1d=80;
    lambda2d_wie=20;
    lambda1d_wie=60;
elseif(tran_mode == 2)  %dwt
    lambda2d=50;
    lambda1d=80;
    lambda2d_wie=20;
    lambda1d_wie=60;
end

%block为原始图像块， tran_block为FFT变换且硬阈值截断后的频域系数(频域， 计算距离的时候采用的是变换块)
[block,tran_block,block2row_idx,block2col_idx]=im2block(img_noise,kHard,pHard,lambda_distHard,0);

%bn_r和bn_c为行和列上的图像块个数
bn_r=floor((row-kHard)/pHard)+1;
bn_c=floor((col-kHard)/pHard)+1;
%基础估计的图像
img_basic_sum=zeros(row,col);
img_basic_weight=zeros(row,col);
%basic处理
fprintf('BM3D: First Stage Start...\n');
%对每个块遍历
for i=1:bn_r
    for j=1:bn_c
        [sim_blk,sim_num,sim_blk_idx]=search_similar_block(i,j,block,tran_block,floor(nHard/pHard),bn_r,bn_c,tauHard,NHard);
        % 协同滤波： 公式(2)
        tran3d_blk_shrink=transform_3d(sim_blk,tran_mode,lambda2d,lambda1d);
        % 聚合： 公式(3)中的说明
        NHard_P=nnz(tran3d_blk_shrink);
        if(NHard_P >1)
            wHard_P=1/NHard_P;
        else
            wHard_P=1;
        end
        blk_est =inv_transform_3d(tran3d_blk_shrink,tran_mode);
        blk_est=real(blk_est);
        
        % 公式(3)
        for k=1:sim_num
            idx=sim_blk_idx(k);
            ir=block2row_idx(idx);
            jr=block2col_idx(idx);
            img_basic_sum(ir:ir+kHard-1,jr:jr+kHard-1) = img_basic_sum(ir:ir+kHard-1,jr:jr+kHard-1) + wHard_P*blk_est(:,:,k);
            img_basic_weight(ir:ir+kHard-1,jr:jr+kHard-1) = img_basic_weight(ir:ir+kHard-1,jr:jr+kHard-1) + wHard_P;
        end
    end
end
fprintf('BM3D: First Stage End...\n');
img_basic=img_basic_sum./img_basic_weight;

if isDisplay
    figure;
    imshow(img_basic,[]);
    title('BM3D:Fist Stage Result');
end


[block_basic,tran_block_basic,block2row_idx_basic,block2col_idx_basic] = im2block(img_basic,kWien,pWien,lambda_distWien,0);
bn_r=floor((row-kWien)/pWien)+1;
bn_c=floor((col-kWien)/pWien)+1;
img_wien_sum=zeros(row,col);
img_wien_weight=zeros(row,col);

fprintf('BM3D: Second Stage Start...\n');
for i=1:1:bn_r
    for j=1:1:bn_c
        % 公式(5)
        [sim_blk_basic,sim_num,sim_blk_basic_idx] = search_similar_block(i,j,block_basic,tran_block_basic,floor(nWien/pWien),bn_r,bn_c,tauWien,NWien);
        % 公式(6)
        tran3d_blk_basic = transform_3d(sim_blk_basic,tran_mode,lambda2d_wie,lambda1d_wie);
        omega_P=(tran3d_blk_basic.^2)./((tran3d_blk_basic.^2)+sigma2);
        % 公式(7)
        tran3d_blk = transform_3d(block(:,:,sim_blk_basic_idx),tran_mode,lambda2d_wie,lambda1d_wie);
        blk_est=inv_transform_3d(omega_P.*tran3d_blk,tran_mode);
        blk_est=real(blk_est);
        NWien_P=nnz(omega_P);
        if(NWien_P >1)
            wWien_P=1/(NWien_P);
        else
            wWien_P=1;
        end
        % 公式(8)
        for k=1:sim_num
            idx=sim_blk_basic_idx(k);
            ir=block2row_idx_basic(idx);
            jr=block2col_idx_basic(idx);
            img_wien_sum(ir:ir+kWien-1,jr:jr+kWien-1) = img_wien_sum(ir:ir+kWien-1,jr:jr+kWien-1) + wWien_P*blk_est(:,:,k);
            img_wien_weight(ir:ir+kWien-1,jr:jr+kWien-1) = img_wien_weight(ir:ir+kWien-1,jr:jr+kWien-1) + wWien_P;
        end
    end
end
fprintf('BM3D: Second Stage End\n');

img_denoise = img_wien_sum./img_wien_weight;
end

function [block,transform_block,block2row_idx,block2col_idx] =im2block(img,k,p,lambda2D,delta)
% 实现图像分块
% Inputs：
%        k: 块大小
%        p: 块移动步长
%        lambda_2D: 收缩阈值
%        delta: 收缩阈值
%  Outputs:
%        block: 返回的块
%        transform_block: 变换后的块
%        block2row_idx: 块索引与图像块的左上角行坐标对应关系
%        block2col_idx: 块索引与图像块的左上角列坐标对应关系
%
[row,col] = size(img);
% 频域去噪中的硬阈值，实际上原文中，对于噪声方差小于40时thres = 0, 具体见公式(1)的说明第2点(即距离计算)
thres = lambda2D*delta*sqrt(2*log(row*col));
% r_num 和 c_num分别表示行和列上可以采集的块的数目
r_num = floor((row-k)/p)+1;
c_num = floor((col-k)/p)+1;
block = zeros(k,k,r_num*c_num);
block2row_idx = [];
block2col_idx = [];
cnt = 1;
for i = 0:r_num-1
    rs = 1+i*p;
    for j = 0:c_num-1
        cs = 1+j*p;
        block(:,:,cnt) = img(rs:rs+k-1,cs:cs+k-1);
        block2row_idx(cnt) = rs;
        block2col_idx(cnt) = cs;
        tr_b = fft2(block(:,:,cnt));
        idx = find(abs(tr_b)<thres);
        tr_b(idx) = 0;
        transform_block(:,:,cnt) = tr_b;
        cnt = cnt+1;
    end
end
end


function [blk_est]=inv_transform_3d(blk_tran3d,tran_mode)
% 3D 逆变换
% Inputs:
%       blk_tran3d: 在频域中，硬阈值滤波的图像块
%       tran_mode: 变换方法
% Outputs:
%       blk_est:
%
global blk_tran1d_s;
global blk_2d_s;
[m,n,blk_num]=size(blk_tran3d);

blk_invtran1d=zeros(m,n,blk_num);
blk_est=zeros(m,n,blk_num);

if(tran_mode==0)    %fft
    for i=1:1:m
        for j=1:1:n
            blk_invtran1d(i,j,:)=ifft(blk_tran3d(i,j,:));
        end
    end
    for i=1:1:blk_num
        blk_est(:,:,i)=ifft2(blk_invtran1d(:,:,i));
    end
elseif(tran_mode==1)  %dct
    for i=1:1:m
        for j=1:1:n
            blk_invtran1d(i,j,:)=idct(blk_tran3d(i,j,:));
        end
    end
    for i=1:1:blk_num
        blk_est(:,:,i)=idct2(blk_invtran1d(:,:,i));
    end
elseif(tran_mode==2)    %dwt
    for i=1:1:m
        for j=1:1:n
            blk_tmp = blk_tran3d(i,j,:);
            blk_invtran1d(i,j,:)=my_ihaar1(blk_tmp);
        end
    end
    for i=1:1:blk_num
        blk_tmp = blk_invtran1d(:,:,i);
        blk_est(:,:,i)=my_ihaar2(blk_tmp);
    end
%     blk_num=length(blk_2d_s);
%     blk_c=waverec2(blk_tran3d,blk_tran1d_s,'haar');
%     blk_est=[];
%     for i=1:1:blk_num
%         blk_est(:,:,i)=waverec2(blk_c(:,i),blk_2d_s{i},'Bior1.5');
%     end
    
else
    error('tran_mode error');
end

end

function blk_tran3d = transform_3d(blk_3d,tran_mode,lambda2d,lambda1d)
% 进行3D变换，即Collaborative Filtering: 在图像块内进行2D变换，在图像块间进行1D变换
% 公式(2)
% Inputs:
%        blk_3d:
%        tran_mode:
% Ouputs:
%
global blk_tran1d_s;
global blk_2d_s;
[m,n,blk_num]=size(blk_3d);

blk_2d_shrink=zeros(m,n,blk_num);
blk_1d_shrink=zeros(m,n,blk_num);

if(tran_mode==0)    %fft
    for i=1:1:blk_num
        blk_tran2d = fft2(blk_3d(:,:,i));
        blk_2d_shrink(:,:,i) = thres_shrink(blk_tran2d,lambda2d);
    end
    for i=1:1:m
        for j=1:1:n
            blk_tran1d = fft(blk_2d_shrink(i,j,:));
            blk_1d_shrink(i,j,:) = thres_shrink(blk_tran1d,lambda1d);
        end
    end
    blk_tran3d=blk_1d_shrink;
    
elseif(tran_mode==1)  %dct
    for i=1:1:blk_num
        blk_tran2d=dct2(blk_3d(:,:,i));
        blk_2d_shrink(:,:,i)=thres_shrink(blk_tran2d,lambda2d);
    end
    for i=1:1:m
        for j=1:1:n
            blk_tran1d=dct(blk_2d_shrink(i,j,:));
            blk_1d_shrink(i,j,:)=thres_shrink(blk_tran1d,lambda1d);
        end
    end
    blk_tran3d=blk_1d_shrink;
    
elseif(tran_mode==2)    %dwt
    for i=1:1:blk_num
        blk_tran2d=my_haar2(blk_3d(:,:,i));
        blk_2d_shrink(:,:,i)=thres_shrink(blk_tran2d,lambda2d);
    end
    
    for i=1:1:m
        for j=1:1:n
            blk_tran1d=my_haar1(blk_2d_shrink(i,j,:));
            blk_1d_shrink(i,j,:)=thres_shrink(blk_tran1d,lambda1d);
        end
    end
    blk_tran3d=blk_1d_shrink;

    
else
    error('tran_mode error');
end
end

function [val]=thres_shrink(data,thres)
% 进行阈值截断： 即 data(i) < thres ? data(i) = 0 : data(i) = data(i)
% Inputs:
%       data: 阈值截断前的数据
%       thres: 阈值
% Ouputs:
%       val: 阈值截断后的数据
% 
val=data;
idx=find(abs(data)<thres);
val(idx)=0;
end

function [sim_blk,sim_num,sim_blk_idx]=search_similar_block(ik,jk,block,tran_block,np,bn_r,bn_c,tau,max_sim_num)
% 搜索相似块
% Inputs:
%       ik, jk： 待搜索相似块的索引
%       block: 图像块集合
%       tran_block： 图像块FFT硬阈值过滤后的FFT系数
%       k: 图像块大小
%       np: floor(nHard / pHard)， 其中nHard表示图像的搜索区域大小， pHard表示块的移动步长
%       bn_r, bn_c: 图像总的行/列可以采集图像块的数目
%       tau: 图像块相似性判断阈值，见公式(1)
%       max_sim_num: 最多保留相似块的数目
% Ouputs:
%       sim_blk:
%       sim_num:
%       sim_blk_idx:
%
% 搜索窗口的左上角，右下角的块索引
in_s = max(ik-floor(np/2),1);
jn_s = max(jk-floor(np/2),1);
in_e = min(ik+floor(np/2),bn_r);
jn_e = min(jk+floor(np/2),bn_c);
% 当前参考块
ref_blk = tran_block(:,:,((ik-1)*bn_c+jk));
ii = in_s:1:in_e;
jj = jn_s:1:jn_e;
[II,JJ] = meshgrid(ii,jj);
IDX = (II-1)*bn_c+JJ;
blk_idx=IDX(:);
% 收缩范围内的全部图像块
cur_blk=tran_block(:,:,blk_idx);
cnt=size(cur_blk,3);
ref_blk_mat=repmat(ref_blk,[1,1,cnt]);
delta_blk=cur_blk-ref_blk_mat;
dist=sum(sum(delta_blk.*delta_blk,1),2);
[dist_sort,dist_idx]=sort(dist);
% 最大相似块是真实相似块和目标参数相似块的最小值
max_num=min(cnt,max_sim_num);
if(dist_sort(max_num)<tau)
    sim_num=max_num;
else
    sim_num=sum(dist_sort(1:max_num)<tau);
end
cnt_idx=dist_idx(1:sim_num);
sim_blk_idx=blk_idx(cnt_idx);
sim_blk=block(:,:,sim_blk_idx);
end
