%% 伪彩色生成
I = imread('Lena512.png');

light = double(I);
I = (double(light))/255.0;

H = 2*pi*(double(light)/255.0);
light(light <= 127) = - light(light <= 127)*(1.5);
light(light > 127) = (255 - light(light > 127)) * 1.5;
S = double(abs(light));

R = zeros(size(H));
G = R;
B = R;
% ɫ��[0,2*pi/3)��Χ�ڶ�Ӧ��->��
index = find(0<=H & H<2*pi/3);
B(index) = I(index).*(1-S(index));
R(index) = I(index).*(1+(S(index).*cos(H(index)))./cos(pi/3-H(index)));
G(index) = 3*I(index)-(R(index)+B(index));
% ɫ��[2*pi/3,4*pi/3)��Ӧ��->��
index = find(2*pi/3<=H & H<4*pi/3);
H(index) = H(index)-2*pi/3;
R(index) = I(index).*(1-S(index));
G(index) = I(index).*(1+(S(index).*cos(H(index)))./cos(pi/3-H(index)));
B(index) = 3*I(index)-(R(index)+G(index));
% ɫ��[4*pi/3,2*pi]��Ӧ��->��
index = find(4*pi/3<=H & H<=2*pi);
H(index) = H(index)-4*pi/3;
G(index) = I(index).*(1-S(index));
B(index) = I(index).*(1+(S(index).*cos(H(index)))./cos(pi/3-H(index)));
R(index) = 3*I(index)-(B(index)+G(index));
% ������ͨ����ΧΪ[0,255]
out = 255*cat(3,R,G,B);
out = uint8(out);
imwrite(out,'./color_lenna.png');
% light=rgb2ycbcr(light);
% light = light(:,:,1);
% light = ind2rgb(light, summer(255));
% imshow(light)
%imwrite(light,'C:\Users\John Wang\Desktop\pot\72.png');
