%% RGBͼֱ��ͼ������������
clear
clc
I=imread('20_1.jpg');%��ȡͼ��
Imatch=imread('100_1.jpg');%��ȡ�ο�ͼ��
R=I(:,:,1);%��ȡԭͼ��Rͨ��
G=I(:,:,2);%��ȡԭͼ��Gͨ��
B=I(:,:,3);%��ȡԭͼ��Bͨ��
Rmatch=Imatch(:,:,1);%��ȡ�ο�ͼ��Rͨ��
Gmatch=Imatch(:,:,2);%��ȡ�ο�ͼ��Gͨ��
Bmatch=Imatch(:,:,3);%��ȡ�ο�ͼ��Bͨ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rmatch_hist=my_imhist(Rmatch);%��ȡ�ο�ͼ��Rͨ��ֱ��ͼ
Gmatch_hist=my_imhist(Gmatch);%��ȡ�ο�ͼ��Gͨ��ֱ��ͼ
Bmatch_hist=my_imhist(Bmatch);%��ȡ�ο�ͼ��Bͨ��ֱ��ͼ

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
Rout=my_histeq(R,Rmatch_hist);%Rͨ��ֱ��ͼƥ��
Gout=my_histeq(G,Gmatch_hist);%Gͨ��ֱ��ͼƥ��
Bout=my_histeq(B,Bmatch_hist);%Bͨ��ֱ��ͼƥ��

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
J(:,:,1)=Rout;
J(:,:,2)=Gout;
J(:,:,3)=Bout;
figure;%��ʾԭͼ�񡢲ο�ͼ���ƥ����ͼ��
subplot(1,3,1),imshow(I);title('ԭͼ��');
%imwrite(I, '../rep1/img/100_sjtu.jpg')
subplot(1,3,2),imshow(Imatch);title('�ο�ͼ��');
%imwrite(Imatch, '../rep1/img/sjtu.jpg')
subplot(1,3,3),imshow(J);title('�涨����ͼ��');
%imwrite(J, '../rep1/img/100_after_sjtu.jpg')
figure;%��ʾԭͼ�񡢲ο�ͼ��͹涨����ͼ���ֱ��ͼ
subplot(3,3,1),imhist(R,64);title('\fontsize{10}ԭͼ��Rͨ��ֱ��ͼ');
subplot(3,3,2),imhist(G,64);title('\fontsize{10}ԭͼ��Gͨ��ֱ��ͼ');
subplot(3,3,3),imhist(B,64);title('\fontsize{10}ԭͼ��Bͨ��ֱ��ͼ');
 
subplot(3,3,4),imhist(Rmatch,64);title('\fontsize{10}�ο�ͼ��Rͨ��ֱ��ͼ');
subplot(3,3,5),imhist(Gmatch,64);title('\fontsize{10}�ο�ͼ��Gͨ��ֱ��ͼ');
subplot(3,3,6),imhist(Bmatch,64);title('\fontsize{10}�ο�ͼ��Bͨ��ֱ��ͼ');
 
subplot(3,3,7),imhist(Rout,64);title('\fontsize{10}�涨����ͼ��Rͨ��ֱ��ͼ');
subplot(3,3,8),imhist(Gout,64);title('\fontsize{10}�涨����ͼ��Gͨ��ֱ��ͼ');
subplot(3,3,9),imhist(Bout,64);title('\fontsize{10}�涨����ͼ��Bͨ��ֱ��ͼ');
%figure.save('../rep1/img/histrgb.png')