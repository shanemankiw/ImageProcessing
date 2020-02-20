function [EdgeLarge,EdgeBetween] = biThreshold(K, highTh, lowTh)
[height, width] = size(K);
K = K/max(max(K)); %��һ��
EdgeLarge = zeros(height,width);%��¼���Ե  
EdgeBetween = zeros(height,width);%��¼���ܵı�Ե��  
for i = 1:height  
    for j = 1:width  
        if K(i,j) >= highTh%С��С��ֵ��������Ϊ��Ե��  
            EdgeLarge(i,j) = K(i,j);  
        else if K(i,j) >= lowTh  
                EdgeBetween(i,j) = K(i,j);  
            end  
        end  
    end  
end