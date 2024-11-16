function [Y, realpercent] = getPartialLabel(Y, percent, bQuiet)  %
obsTarget_index = zeros(size(Y));  % �ѹ۲����ݾ���obsTarget_index����ʼ��Ϊȫ��
totoalNum = sum(sum(Y ~= 0));  % ����Y�з���Ԫ�ص�����
totoalAddNum = 0;
[N, ~] = size(Y);
realpercent = 0;
maxIteration = 50;
factor = 2;
count = 0;
if percent > 0
    while realpercent < percent
        if maxIteration == 0  % ��������
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
            index = find(Y(i, :) ~= 1);  % �ҵ���i���б�ǩ������1������
            if length(index) >= factor
                addNum = round(rand*(length(index)));  % �������һ������0�ͷ����ǩ����֮�������addNum
                totoalAddNum = totoalAddNum + addNum;
                realpercent = totoalAddNum/totoalNum;
                if addNum > 0
                    index = index(randperm(length(index)));  % �����������˳��
                    Y(i, index(1:addNum)) = 1;  % ѡ��addNum�����ı�ǩ��Ϊ1����Ϊģ��ƫ�������
                    obsTarget_index(i, index(1:addNum))= 1;  % ��obsTarget_index�еĶ�Ӧλ�ñ��Ϊ1
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
