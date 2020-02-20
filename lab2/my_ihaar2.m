%% 二维哈尔逆变换
function array_2d=my_ihaar2(array_2d)

[h, w]= size(array_2d);
len_index = min(ceil(log2(h)), ceil(log2(w)));

indexes = zeros([len_index, 2]);

i = 0;
%% 把index放进去准备好
while((h~=1)&&(w~=1))
    indexes(len_index - i,:) = [h,w];
    i = i + 1;
    h = round(h/2);
    w = round(w/2);
end

%% 二维哈尔逆变换
for i=1:len_index
    h = indexes(i,1);
    w = indexes(i,2);
    new = my_ihaar_base(array_2d(1:h, 1:w));
    array_2d(1:h, 1:w) = new;
    array_2d = permute(array_2d, [2 1]);
    new = my_ihaar_base(array_2d(1:w, 1:h));
    array_2d(1:w, 1:h) = new;
    array_2d = permute(array_2d, [2 1]);
end