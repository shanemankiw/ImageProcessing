%% 一维哈尔变换
function new=my_haar1(array_1d)

new = zeros(size(array_1d));
shape = size(array_1d);
dim = ndims(array_1d);
length = shape(dim);
reshape(array_1d, [1,length]);
if (mod(length,2)~=0)
    length = length + 1;
end

length = length / 2;

for n = 1:length
    new(n) = 0.5*(array_1d(2*n-1) + array_1d(2*n));
    new(n + length) = 0.5*(array_1d(2*n-1) - array_1d(2*n));
    
end