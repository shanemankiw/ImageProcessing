%% 直方图特例化（单通道）
function eq_image = my_histeq(image, match_hist)
eq_image = image;
origin_hist = my_imhist(image);
trans_array = zeros(1,256);     % 转换对应向量trans_array(2) = 5 表示原图中灰度值2（其实是1）对应到新图像灰度值5
ko = cumsum(origin_hist);       % 原图直方图累加
km = cumsum(match_hist);        % 匹配图直方图累加
j = 1;
for i = 1:256
    while ko(i) > km(j) && j < 256 % 找右界
        j = j+1;
    end
    trans_array(i) = j;
    eq_image(find(image == i-1)) = j-1;     %注意matlab索引开始为1，灰度值开始为0
end
% imshow(eq_image)
end