function [Gr, Grx, Gry] = sobel_dif(G)
[height, width] = size(G);
% Sobel算子
sobelx = [-1, 0, 1;
        -2, 0, 2;
        -1, 0, 1];
sobely = [-1, -2, -1;
        0, 0, 0;
        1, 2, 1];
Grx = zeros(size(G));
Gry = Grx;
Gr = Grx;
for i = 1:height  
    for j = 1:width  
        sumx = 0;%临时变量
        sumy = 0;
        for k = 1:3  
            for m = 1:3  
                if (i-2+k) > 0 && (i -2+ k) <= height && (j -2+ m) > 0 && (j -2+ m) < width  
                    sumx = sumx + sobelx(k,m) * G(i -2+ k,j -2+ m);
                    sumy = sumy + sobely(k,m) * G(i -2+ k,j -2+ m);
                end  
            end  
        end  
        Grx(i,j) = sumx;
        Gry(i,j) = sumy;
        Gr(i,j) = sqrt(sumx*sumx + sumy*sumy);
    end  
end