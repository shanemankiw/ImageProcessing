function K = NMS(Gr, Grx, Gry)

[height, width] = size(Gr);
K = Gr;%记录进行非极大值抑制后的梯度  

%设置图像边缘为不可能的边缘点 
for j = 1:width  
    K(1,j) = 0;  
    K(height,j) = 0; 
end   
for i = 2:width - 1  
    K(i,1) = 0;  
    K(i,width) = 0; 
end  

%% 稍作修改   
for i = 2:height - 1  
    for j = 2:width - 1  
        %当前像素点的梯度值为0，则一定不是边缘点  
        if Gr(i,j) == 0  
            K(i,j) = 0;  
        else  
            gradX = Grx(i,j);%当前点x方向导数  
            gradY = Gry(i,j);%当前点y方向导数  
            gradTemp = Gr(i,j);%当前点梯度  
            %如果Y方向幅度值较大  
            if abs(gradY) > abs(gradX)  
                weight = abs(gradX) ./ abs(gradY);
                grad2 = Gr(i - 1,j);  
                grad4 = Gr(i + 1,j);  
                %如果x、y方向导数符号相同  
                %像素点位置关系  
                %g1 g2  
                %   C  
                %   g4 g3  
                if gradX * gradY > 0  
                    grad1 = Gr(i - 1,j - 1);  
                    grad3 = Gr(i + 1,j + 1);  
                else  
                    %如果x、y方向导数符号反  
                    %像素点位置关系  
                    %   g2 g1  
                    %   C  
                    %g3 g4  
                    grad1 = Gr(i - 1,j + 1);  
                    grad3 = Gr(i + 1,j - 1);  
                end
%                 grad_2 = [grad2, 0];
%                 grad_4 = [grad4, 0];
%                 grad_3 = [grad3, grad3];
%                 grad_1 = [grad1, grad1];
            %如果X方向幅度值较大  
            else  
                weight = abs(gradY) ./ abs(gradX);  
                grad2 = Gr(i,j - 1);  
                grad4 = Gr(i,j + 1);  
                %如果x、y方向导数符号相同  
                %像素点位置关系  
                %g3  
                %g4 C g2  
                %     g1  
                if gradX * gradY > 0  
                    grad1 = Gr(i + 1,j + 1);  
                    grad3 = Gr(i - 1,j - 1);  
                else  
                    %如果x、y方向导数符号反  
                    %像素点位置关系  
                    %     g1  
                    %g4 C g2  
                    %g3  
                    grad1 = Gr(i - 1,j + 1);  
                    grad3 = Gr(i + 1,j - 1);
                end 
%                 grad_2 = [0, grad2];
%                 grad_4 = [0, grad4];
%                 grad_3 = [grad3, grad3];
%                 grad_1 = [grad1, grad1];
            end  
            %利用grad1-grad4对梯度插值
            %dir = [abs(gradX), abs(gradY)] ./ sqrt(gradX*gradX + gradY*gradY);
            
            % 这里我尝试了另一种方法，用向量点积进行加权
            % 但是最后没有成功，分析在文章中有
            %gradTemp1 = dir * grad_1' + dir * grad_2';
            %gradTemp2 = dir * grad_3' + dir * grad_4';
            
            gradTemp1 = weight * grad1 + (1 - weight) * grad2;  
            gradTemp2 = weight * grad3 + (1 - weight) * grad4;  
            %当前像素的梯度是局部的最大值，可能是边缘点  
            if gradTemp >= gradTemp1 && gradTemp >= gradTemp2  
                K(i,j) = gradTemp;  
            else  
                %不可能是边缘点  
                K(i,j) = 0;  
            end  
        end  
    end  
end
