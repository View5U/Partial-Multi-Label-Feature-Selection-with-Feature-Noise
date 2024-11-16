function [Y, realpercent] = getPartialLabel(Y, percent, bQuiet)  %
obsTarget_index = zeros(size(Y));  % 已观察数据矩阵obsTarget_index，初始化为全零
totoalNum = sum(sum(Y ~= 0));  % 计算Y中非零元素的总数
totoalAddNum = 0;
[N, ~] = size(Y);
realpercent = 0;
maxIteration = 50;
factor = 2;
count = 0;
if percent > 0
    while realpercent < percent
        if maxIteration == 0  % 控制速率
            factor = 1;
            maxIteration = 10;
            if count == 1
                break;
            end
            count = count + 1;
        else
            maxIteration = maxIteration - 1;
        end  % if maxIteration == 0
        for i = 1:N
            index = find(Y(i, :) ~= 1);  % 找到第i行中标签不等于1的索引
            if length(index) >= factor
                addNum = round(rand*(length(index)));  % 随机生成一个介于0和非零标签数量之间的整数addNum
                totoalAddNum = totoalAddNum + addNum;
                realpercent = totoalAddNum/totoalNum;
                if addNum > 0
                    index = index(randperm(length(index)));  % 随机打乱索引顺序
                    Y(i, index(1:addNum)) = 1;  % 选择addNum数量的标签改为1，即为模拟偏标记数据
                    obsTarget_index(i, index(1:addNum))= 1;  % 将obsTarget_index中的对应位置标记为1
                end
                if realpercent >= percent
                    break;
                end
            end
        end  % for i = 1:N
    end  % while realpercent < percent
end  % if percent > 0

if bQuiet == 0
    fprintf('Totoal Number of Totoal Num : %d\n ', totoalNum);
    fprintf('Number of Totoal Add Num : %d\n ', totoalAddNum);
    fprintf('Given percent/Real percent : %.2f / %.2f\n', percent, totoalAddNum/totoalNum);
end
end
