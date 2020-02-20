%% 一维哈尔逆变换
function array_1d=my_ihaar1(array_1d)

blk_invtran1d = zeros(size(array_1d));
shape = size(array_1d);
length = shape(1);

len_index = ceil(log2(length));
indexes = zeros(len_index);
i = 0;

%% 把index放进去准备好
while(length~=1)
    indexes(len_index - i) = length;
    i = i + 1;
    length = round(length/2);
end

%% 一维哈尔逆变换
for i=1:len_index
    blk_invtran1d = my_ihaar_base(array_1d(1:indexes(i)+1));
    array_1d(1:indexes(i)+1) = blk_invtran1d;
end