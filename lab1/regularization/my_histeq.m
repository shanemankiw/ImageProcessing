%% ֱ��ͼ����������ͨ����
function eq_image = my_histeq(image, match_hist)
eq_image = image;
origin_hist = my_imhist(image);
trans_array = zeros(1,256);     % ת����Ӧ����trans_array(2) = 5 ��ʾԭͼ�лҶ�ֵ2����ʵ��1����Ӧ����ͼ��Ҷ�ֵ5
ko = cumsum(origin_hist);       % ԭͼֱ��ͼ�ۼ�
km = cumsum(match_hist);        % ƥ��ͼֱ��ͼ�ۼ�
j = 1;
for i = 1:256
    while ko(i) > km(j) && j < 256 % ���ҽ�
        j = j+1;
    end
    trans_array(i) = j;
    eq_image(find(image == i-1)) = j-1;     %ע��matlab������ʼΪ1���Ҷ�ֵ��ʼΪ0
end
% imshow(eq_image)
end