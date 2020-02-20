function edge = Connect(EdgeLarge, EdgeBetween)
[height, width] = size(EdgeLarge);

MAXSIZE = 99999;  
queue = zeros(MAXSIZE,2);%模拟队列  
front = 1;%队头  
rear = 1;%队尾  
edge = zeros(height,width);  
for i = 1:height  
    for j = 1:width  
        if EdgeLarge(i,j) > 0  
            %强点入队  
            queue(rear,1) = i;  
            queue(rear,2) = j;  
            rear = rear + 1;  
            edge(i,j) = 1;%EdgeLarge(i,j);  
            EdgeLarge(i,j) = 0;%避免重复计算  
        end  
        while front ~= rear%队不空  
            %队头出队  
            temp_i = queue(front,1);  
            temp_j = queue(front,2);  
            front = front + 1;  
            %8-连通域寻找可能的边缘点  
            %左上方  
            if EdgeBetween(temp_i - 1,temp_j - 1) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i - 1,temp_j - 1) = 1;%K(temp_i - 1,temp_j - 1);  
                EdgeBetween(temp_i - 1,temp_j - 1) = 0;%避免重复计算  
                %入队  
                queue(rear,1) = temp_i - 1;  
                queue(rear,2) = temp_j - 1;  
                rear = rear + 1;  
            end  
            %正上方  
            if EdgeBetween(temp_i - 1,temp_j) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i - 1,temp_j) = 1;%K(temp_i - 1,temp_j);  
                EdgeBetween(temp_i - 1,temp_j) = 0;  
                %入队  
                queue(rear,1) = temp_i - 1;  
                queue(rear,2) = temp_j;  
                rear = rear + 1;  
            end  
            %右上方  
            if EdgeBetween(temp_i - 1,temp_j + 1) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i - 1,temp_j + 1) = 1;%K(temp_i - 1,temp_j + 1);  
                EdgeBetween(temp_i - 1,temp_j + 1) = 0;  
                %入队  
                queue(rear,1) = temp_i - 1;  
                queue(rear,2) = temp_j + 1;  
                rear = rear + 1;  
            end  
            %正左方  
            if EdgeBetween(temp_i,temp_j - 1) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i,temp_j - 1) = 1;%K(temp_i,temp_j - 1);  
                EdgeBetween(temp_i,temp_j - 1) = 0;  
                %入队  
                queue(rear,1) = temp_i;  
                queue(rear,2) = temp_j - 1;  
                rear = rear + 1;  
            end  
            %正右方  
            if EdgeBetween(temp_i,temp_j + 1) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i,temp_j + 1) = 1;%K(temp_i,temp_j + 1);  
                EdgeBetween(temp_i,temp_j + 1) = 0;  
                %入队  
                queue(rear,1) = temp_i;  
                queue(rear,2) = temp_j + 1;  
                rear = rear + 1;  
            end  
            %左下方  
            if EdgeBetween(temp_i + 1,temp_j - 1) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i + 1,temp_j - 1) = 1;%K(temp_i + 1,temp_j - 1);  
                EdgeBetween(temp_i + 1,temp_j - 1) = 0;  
                %入队  
                queue(rear,1) = temp_i + 1;  
                queue(rear,2) = temp_j - 1;  
                rear = rear + 1;  
            end  
            %正下方  
            if EdgeBetween(temp_i + 1,temp_j) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i + 1,temp_j) = 1;%K(temp_i + 1,temp_j);  
                EdgeBetween(temp_i + 1,temp_j) = 0;  
                %入队  
                queue(rear,1) = temp_i + 1;  
                queue(rear,2) = temp_j;  
                rear = rear + 1;  
            end  
            %右下方  
            if EdgeBetween(temp_i + 1,temp_j + 1) > 0%把在强点周围的弱点变为强点  
                EdgeLarge(temp_i + 1,temp_j + 1) = 1;%K(temp_i + 1,temp_j + 1);  
                EdgeBetween(temp_i + 1,temp_j + 1) = 0;  
                %入队  
                queue(rear,1) = temp_i + 1;  
                queue(rear,2) = temp_j + 1;  
                rear = rear + 1;  
            end  
        end  
    end  
end