%% 二维哈尔变换
function array_2d=my_haar2(array_2d)

[h, w]= size(array_2d);

%% 把index放进去准备好
while((h~=1)&&(w~=1))
    new = my_haar2_base(array_2d(1:h,1:w));
    array_2d(1:h,1:w) = new;
    h = round(h/2);
    w = round(w/2);
end