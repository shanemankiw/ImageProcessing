function [EdgeLarge,EdgeBetween] = biThreshold(K, highTh, lowTh)
[height, width] = size(K);
K = K/max(max(K)); %归一化
EdgeLarge = zeros(height,width);%记录真边缘  
EdgeBetween = zeros(height,width);%记录可能的边缘点  
for i = 1:height  
    for j = 1:width  
        if K(i,j) >= highTh%小于小阈值，不可能为边缘点  
            EdgeLarge(i,j) = K(i,j);  
        else if K(i,j) >= lowTh  
                EdgeBetween(i,j) = K(i,j);  
            end  
        end  
    end  
end