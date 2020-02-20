%% haar2的基础函数
function new = my_haar2_base(array)
%new = zeros(size(array));

[num_h, num_w] = size(array);

if mod(num_h,2) == 1
    num_h = num_h + 1;
end

if mod(num_w,2) == 1
    num_w = num_w + 1;
end

w = round(num_w / 2);
h = round(num_h / 2);

for x=1:w
    
    new(:,x) = 0.5*(array(:,2*x-1)+array(:, 2*x));
    new(:,w+x) = 0.5*(array(:,2*x-1)-array(:,2*x));
    
end
array = new;
for y=1:h
    
    new(y,:) = 0.5*(array(2*y-1,:)+array(2*y,:));
    new(h+y,:) = 0.5*(array(2*y-1,:)-array(2*y,:));
    
end