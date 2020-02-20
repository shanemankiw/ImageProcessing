function K = NMS(Gr, Grx, Gry)

[height, width] = size(Gr);
K = Gr;%��¼���зǼ���ֵ���ƺ���ݶ�  

%����ͼ���ԵΪ�����ܵı�Ե�� 
for j = 1:width  
    K(1,j) = 0;  
    K(height,j) = 0; 
end   
for i = 2:width - 1  
    K(i,1) = 0;  
    K(i,width) = 0; 
end  

%% �����޸�   
for i = 2:height - 1  
    for j = 2:width - 1  
        %��ǰ���ص���ݶ�ֵΪ0����һ�����Ǳ�Ե��  
        if Gr(i,j) == 0  
            K(i,j) = 0;  
        else  
            gradX = Grx(i,j);%��ǰ��x������  
            gradY = Gry(i,j);%��ǰ��y������  
            gradTemp = Gr(i,j);%��ǰ���ݶ�  
            %���Y�������ֵ�ϴ�  
            if abs(gradY) > abs(gradX)  
                weight = abs(gradX) ./ abs(gradY);
                grad2 = Gr(i - 1,j);  
                grad4 = Gr(i + 1,j);  
                %���x��y������������ͬ  
                %���ص�λ�ù�ϵ  
                %g1 g2  
                %   C  
                %   g4 g3  
                if gradX * gradY > 0  
                    grad1 = Gr(i - 1,j - 1);  
                    grad3 = Gr(i + 1,j + 1);  
                else  
                    %���x��y���������ŷ�  
                    %���ص�λ�ù�ϵ  
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
            %���X�������ֵ�ϴ�  
            else  
                weight = abs(gradY) ./ abs(gradX);  
                grad2 = Gr(i,j - 1);  
                grad4 = Gr(i,j + 1);  
                %���x��y������������ͬ  
                %���ص�λ�ù�ϵ  
                %g3  
                %g4 C g2  
                %     g1  
                if gradX * gradY > 0  
                    grad1 = Gr(i + 1,j + 1);  
                    grad3 = Gr(i - 1,j - 1);  
                else  
                    %���x��y���������ŷ�  
                    %���ص�λ�ù�ϵ  
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
            %����grad1-grad4���ݶȲ�ֵ
            %dir = [abs(gradX), abs(gradY)] ./ sqrt(gradX*gradX + gradY*gradY);
            
            % �����ҳ�������һ�ַ�����������������м�Ȩ
            % �������û�гɹ�����������������
            %gradTemp1 = dir * grad_1' + dir * grad_2';
            %gradTemp2 = dir * grad_3' + dir * grad_4';
            
            gradTemp1 = weight * grad1 + (1 - weight) * grad2;  
            gradTemp2 = weight * grad3 + (1 - weight) * grad4;  
            %��ǰ���ص��ݶ��Ǿֲ������ֵ�������Ǳ�Ե��  
            if gradTemp >= gradTemp1 && gradTemp >= gradTemp2  
                K(i,j) = gradTemp;  
            else  
                %�������Ǳ�Ե��  
                K(i,j) = 0;  
            end  
        end  
    end  
end
