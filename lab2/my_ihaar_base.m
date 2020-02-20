%% 哈尔变换逆变换的基本运算
function blk_invtran1d=my_ihaar_base(pic_1d)

blk_invtran1d = zeros(size(pic_1d));
shape= size(pic_1d);
length_ori = shape(1);

% 注意这里判断一下维数
aMatrix = size(shape);
dimention = aMatrix(2);
if dimention == 1
    if (mod(length_ori)~=0)
        length_ori = length_ori + 1;
    end
    
    length = length_ori / 2;
    for n = 1:length
        blk_invtran1d(2*n - 1) = pic_1d(n)+pic_1d(n+length);
        blk_invtran1d(2*n) = pic_1d(n)-pic_1d(n+length);
    end
    
    
elseif dimention == 2
    if (mod(length_ori, 2)~=0)
        length_ori = length_ori + 1;
    end
    
    length = length_ori / 2;
    for n = 1:length
        blk_invtran1d(2*n-1,:) = pic_1d(n,:)+pic_1d(n+length,:);
        blk_invtran1d(2*n,:) = pic_1d(n,:)-pic_1d(n+length,:);
    end
end