function G = Gaussian_filter(I, N, sigma)
[height, width] = size(I);
kernel = zeros(N,N);%��˹�����
middle = (N-1)/2;
for i = 1:N 
    for j = 1:N  
        kernel(i,j) = exp((-(i-middle-1)*(i-middle-1)-(j-middle-1)*(j-middle-1))/(2*sigma*sigma))/(2*3.14*sigma*sigma);%��˹��ʽ  
    end  
end  
kernel = kernel./sum(sum(kernel));%��׼��  
  
%��ͼ��ʵʩ��˹�˲�  
for i = 1:height  
    for j = 1:width  
        a_sum = 0;%��ʱ����  
        for k = 1:N  
            for m = 1:N  
                if (i-middle-1+k) > 0 && (i -middle-1+ k) <= height && (j -middle-1+ m) > 0 && (j -middle-1+ m) < width  
                    a_sum = a_sum + kernel(k,m) * I(i -middle-1+ k,j -middle-1+ m);  
                end  
            end  
        end  
        G(i,j) = a_sum;  
    end  
end