%% ֱ��ͼ�Ŀ��ӻ�
function grayCount = my_imhist(image)
[m,n] = size(image);
uimage = uint8(image);      % ȡ��
grayCount = zeros(1,256);   %�Ҷ�������ע���һ��λ�÷ŵ��ǻҶ�0�ĸ���
for i = 1:m
    for j = 1:n
         grayCount(1,uimage(i,j)+1) = grayCount(1,uimage(i,j)+1)+1;
    end
end
stem(grayCount,'.');
end