function edge = Connect(EdgeLarge, EdgeBetween)
[height, width] = size(EdgeLarge);

MAXSIZE = 99999;  
queue = zeros(MAXSIZE,2);%ģ�����  
front = 1;%��ͷ  
rear = 1;%��β  
edge = zeros(height,width);  
for i = 1:height  
    for j = 1:width  
        if EdgeLarge(i,j) > 0  
            %ǿ�����  
            queue(rear,1) = i;  
            queue(rear,2) = j;  
            rear = rear + 1;  
            edge(i,j) = 1;%EdgeLarge(i,j);  
            EdgeLarge(i,j) = 0;%�����ظ�����  
        end  
        while front ~= rear%�Ӳ���  
            %��ͷ����  
            temp_i = queue(front,1);  
            temp_j = queue(front,2);  
            front = front + 1;  
            %8-��ͨ��Ѱ�ҿ��ܵı�Ե��  
            %���Ϸ�  
            if EdgeBetween(temp_i - 1,temp_j - 1) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i - 1,temp_j - 1) = 1;%K(temp_i - 1,temp_j - 1);  
                EdgeBetween(temp_i - 1,temp_j - 1) = 0;%�����ظ�����  
                %���  
                queue(rear,1) = temp_i - 1;  
                queue(rear,2) = temp_j - 1;  
                rear = rear + 1;  
            end  
            %���Ϸ�  
            if EdgeBetween(temp_i - 1,temp_j) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i - 1,temp_j) = 1;%K(temp_i - 1,temp_j);  
                EdgeBetween(temp_i - 1,temp_j) = 0;  
                %���  
                queue(rear,1) = temp_i - 1;  
                queue(rear,2) = temp_j;  
                rear = rear + 1;  
            end  
            %���Ϸ�  
            if EdgeBetween(temp_i - 1,temp_j + 1) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i - 1,temp_j + 1) = 1;%K(temp_i - 1,temp_j + 1);  
                EdgeBetween(temp_i - 1,temp_j + 1) = 0;  
                %���  
                queue(rear,1) = temp_i - 1;  
                queue(rear,2) = temp_j + 1;  
                rear = rear + 1;  
            end  
            %����  
            if EdgeBetween(temp_i,temp_j - 1) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i,temp_j - 1) = 1;%K(temp_i,temp_j - 1);  
                EdgeBetween(temp_i,temp_j - 1) = 0;  
                %���  
                queue(rear,1) = temp_i;  
                queue(rear,2) = temp_j - 1;  
                rear = rear + 1;  
            end  
            %���ҷ�  
            if EdgeBetween(temp_i,temp_j + 1) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i,temp_j + 1) = 1;%K(temp_i,temp_j + 1);  
                EdgeBetween(temp_i,temp_j + 1) = 0;  
                %���  
                queue(rear,1) = temp_i;  
                queue(rear,2) = temp_j + 1;  
                rear = rear + 1;  
            end  
            %���·�  
            if EdgeBetween(temp_i + 1,temp_j - 1) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i + 1,temp_j - 1) = 1;%K(temp_i + 1,temp_j - 1);  
                EdgeBetween(temp_i + 1,temp_j - 1) = 0;  
                %���  
                queue(rear,1) = temp_i + 1;  
                queue(rear,2) = temp_j - 1;  
                rear = rear + 1;  
            end  
            %���·�  
            if EdgeBetween(temp_i + 1,temp_j) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i + 1,temp_j) = 1;%K(temp_i + 1,temp_j);  
                EdgeBetween(temp_i + 1,temp_j) = 0;  
                %���  
                queue(rear,1) = temp_i + 1;  
                queue(rear,2) = temp_j;  
                rear = rear + 1;  
            end  
            %���·�  
            if EdgeBetween(temp_i + 1,temp_j + 1) > 0%����ǿ����Χ�������Ϊǿ��  
                EdgeLarge(temp_i + 1,temp_j + 1) = 1;%K(temp_i + 1,temp_j + 1);  
                EdgeBetween(temp_i + 1,temp_j + 1) = 0;  
                %���  
                queue(rear,1) = temp_i + 1;  
                queue(rear,2) = temp_j + 1;  
                rear = rear + 1;  
            end  
        end  
    end  
end