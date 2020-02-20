%% RGB图直方图特例化主函数
clear
clc
I=imread('20_1.jpg');%读取图像
Imatch=imread('100_1.jpg');%读取参考图像
R=I(:,:,1);%获取原图像R通道
G=I(:,:,2);%获取原图像G通道
B=I(:,:,3);%获取原图像B通道
Rmatch=Imatch(:,:,1);%获取参考图像R通道
Gmatch=Imatch(:,:,2);%获取参考图像G通道
Bmatch=Imatch(:,:,3);%获取参考图像B通道

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rmatch_hist=my_imhist(Rmatch);%获取参考图像R通道直方图
Gmatch_hist=my_imhist(Gmatch);%获取参考图像G通道直方图
Bmatch_hist=my_imhist(Bmatch);%获取参考图像B通道直方图

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rout=my_histeq(R,Rmatch_hist);%R通道直方图匹配
Gout=my_histeq(G,Gmatch_hist);%G通道直方图匹配
Bout=my_histeq(B,Bmatch_hist);%B通道直方图匹配

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J(:,:,1)=Rout;
J(:,:,2)=Gout;
J(:,:,3)=Bout;
figure;%显示原图像、参考图像和匹配后的图像
subplot(1,3,1),imshow(I);title('原图像');
%imwrite(I, '../rep1/img/100_sjtu.jpg')
subplot(1,3,2),imshow(Imatch);title('参考图像');
%imwrite(Imatch, '../rep1/img/sjtu.jpg')
subplot(1,3,3),imshow(J);title('规定化后图像');
%imwrite(J, '../rep1/img/100_after_sjtu.jpg')
figure;%显示原图像、参考图像和规定化后图像的直方图
subplot(3,3,1),imhist(R,64);title('\fontsize{10}原图像R通道直方图');
subplot(3,3,2),imhist(G,64);title('\fontsize{10}原图像G通道直方图');
subplot(3,3,3),imhist(B,64);title('\fontsize{10}原图像B通道直方图');
 
subplot(3,3,4),imhist(Rmatch,64);title('\fontsize{10}参考图像R通道直方图');
subplot(3,3,5),imhist(Gmatch,64);title('\fontsize{10}参考图像G通道直方图');
subplot(3,3,6),imhist(Bmatch,64);title('\fontsize{10}参考图像B通道直方图');
 
subplot(3,3,7),imhist(Rout,64);title('\fontsize{10}规定化后图像R通道直方图');
subplot(3,3,8),imhist(Gout,64);title('\fontsize{10}规定化后图像G通道直方图');
subplot(3,3,9),imhist(Bout,64);title('\fontsize{10}规定化后图像B通道直方图');
%figure.save('../rep1/img/histrgb.png')