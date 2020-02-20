function [Gr, Grx, Gry] = direct_dif(G)
% 直接一阶差分
%相邻的梯度值都很大
[height, width] = size(G);
Grx = zeros(height,width);%x方向梯度  
Gry = zeros(height,width);%y方向梯度  
Gr = zeros(height,width);  
for i = 1:height - 1  
    for j = 1:width - 1  
        Grx(i,j) = G(i,j + 1) - G(i,j);  
        Gry(i,j) = G(i + 1,j) - G(i,j);  
        Gr(i,j) = sqrt(Grx(i,j) * Grx(i,j) + Gry(i,j) * Gry(i,j));  
    end  
end  