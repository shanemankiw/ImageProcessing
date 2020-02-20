%% 直方图的可视化
function grayCount = my_imhist(image)
[m,n] = size(image);
uimage = uint8(image);      % 取整
grayCount = zeros(1,256);   %灰度向量，注意第一个位置放的是灰度0的个数
for i = 1:m
    for j = 1:n
         grayCount(1,uimage(i,j)+1) = grayCount(1,uimage(i,j)+1)+1;
    end
end
stem(grayCount,'.');
end