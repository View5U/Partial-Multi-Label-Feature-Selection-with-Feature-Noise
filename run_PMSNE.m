clear;
data_name = 'birds.mat';
load(data_name);

rng(0);
partialRate  = 1;
feaNoiseRate = 0.3;

evamode = 1;  % 0:single, 1:curve
HyperPara.ablationx = 0;
HyperPara.ablationy = 0;
HyperPara.ablationd = 0;

data = [train_data; test_data];
target = [train_target, test_target];

% target = target';
% target(target==-1)=0;
% N = length(target);
% indices = crossvalind('Kfold',1:N,5);
% test_idxs = (indices == 1);
% train_idxs = ~test_idxs;
% train_data=data(train_idxs,:);train_target=target(train_idxs,:)';
% test_data=data(test_idxs,:);test_target=target(test_idxs,:)';

[train_data, settings]=mapminmax(train_data');
test_data=mapminmax('apply',test_data',settings);
train_data=train_data';
test_data=test_data';
train_data(isnan(train_data))=0;
test_data(isnan(test_data))=0;

train_target(train_target==-1) = 0;
test_target(test_target==-1) = 0;
PL = getPartialLabel(train_target', partialRate, 0);

[num_instance, num_feature] = size(train_data);
[num_label, ~]   = size(train_target);

clean_data = [train_data; test_data];
clean_train_data = train_data;
clean_test_data = test_data;
noisy_data = FeatureNoise(clean_data, feaNoiseRate);
train_data = noisy_data(1:num_instance,:);
test_data = noisy_data(num_instance+1:end,:);

%% run PMSNE 
HyperPara.ins_num = num_instance;
HyperPara.class = size(train_target, 1);
HyperPara.k = 10;
HyperPara.closedform = 0;
HyperPara.uselip = 0;
fprintf('Running--');

HyperPara.alpha   = 8*num_instance/num_feature;     % W21
HyperPara.beta    = 0.5;                            % ADL-D
HyperPara.gamma   = 2;                              % D-S
HyperPara.delta   = 1;                              % Y.*D
HyperPara.epsilon = 4;                              % A1
HyperPara.maxIter = 100;
HyperPara.minLossMargin = 0.001;
[W, A, Distribution] = Optimization_PMSNE(train_data, PL, HyperPara);
% train_data = A*train_data;
W = W(1:end-1,:);

[dumb, index] = sort(sum(W.*W,2),'descend');
Num = 10;
Smooth = 1;
PL(PL==0) = -1;

if evamode == 1
    NumOfInterval = 25;
    Dt = ceil(num_feature*0.5);  % ceil(num_feature*0.5) or 50
    step = ceil(Dt/NumOfInterval); % ceil(Dt/NumOfInterval) or 2
    
    iterResult = zeros(15, NumOfInterval);
    for d = 1:step:((NumOfInterval-1)*step+1)
        order = (d-1)/step+1;
        f = index(1:d);      
        [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f), PL',Num,Smooth);
        [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
            MLKNN_test(train_data(:,f), PL', test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);       
        fprintf('-- Evaluation\n');
        test_target(test_target==0) = -1;
        tmpResult = EvaluationAll(Pre_Labels,Outputs,test_target);
        iterResult(:,order) = iterResult(:,order) + tmpResult;
    end
else
    if num_feature <= 100
        d = ceil(0.4*num_feature);
    elseif num_feature <=500
        d = ceil(0.3*num_feature);
    elseif num_feature <= 1000
        d = ceil(0.2*num_feature);
    else
        d = ceil(0.1*num_feature);
    end
    order = 1;
    iterResult = zeros(15, 1);
    f = index(1:d);
    [Prior,PriorN,Cond,CondN]=MLKNN_train(train_data(:,f),train_data,Num,Smooth);
    [HammingLoss,RankingLoss,Coverage,Average_Precision,macrof1,microf1,Outputs,Pre_Labels]=...
        MLKNN_test(train_data(:,f), train_data, test_data(:,f),test_target,Num,Prior,PriorN,Cond,CondN);
    fprintf('-- Evaluation\n');
    tmpResult = EvaluationAll(Pre_Labels,Outputs,test_target);
    iterResult(:,order) = iterResult(:,order) + tmpResult;
end

Avg_Result      = zeros(15,2);
Avg_Result(:,1) = mean(iterResult,2);
Avg_Result(:,2) = std(iterResult,1,2);

if evamode == 1
    model_name = "PMSNE";
    x = 2:2:50;
    figure;
    hold on;
    plot(x, iterResult(14,:), 'color', '#000000', 'DisplayName', model_name, 'Marker', 'h');
    % axis([0 50 0.05 0.5])
    hold off;
    grid on;
    xlabel('Seleted Features');
    ylabel('rl');
    title(data_name);
    legend('Location', 'best');
end

