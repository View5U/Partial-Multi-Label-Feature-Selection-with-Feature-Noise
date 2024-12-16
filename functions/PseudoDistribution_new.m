function S = PseudoDistribution_new(X, Y, Y_origin, HyperPara)
LP  = LabelProtype(X, Y);
SIL = cosineSimilarity(X, LP, HyperPara);
SIL(isnan(SIL)) = 0;

S = SimplexProj(SIL);
S = AddLogicalLabel(S, Y_origin, HyperPara);
S = SimplexProj(S);

% mean_S = mean(S(:));
% S(S < mean_S) = 0;
end

function protype = LabelProtype(X, Y)
num_label = size(Y, 2);
num_ins = size(X, 1);
num_fea = size(X, 2);
protype = zeros(num_label, num_fea);
card = sum(sum(Y))/num_ins;

for l = 1:num_label
    weight = 0;
    sum_value_l = sum(Y(:, l));
    for i = 1:num_ins
        if sum(Y(i,:)) < card*Inf
            value_il = Y(i, l);
            sum_value_j = sum(Y(i, :));
            weight_i = (value_il^2) / (sum_value_l*sum_value_j);  % weight
            protype(l, :) = protype(l, :) + weight_i .* X(i, :);
            weight = weight + weight_i;
        end
    end
    if sum_value_l == 0
        protype(l, :) = zeros(1, num_fea);
    else
        protype(l, :) = protype(l, :) ./ weight;
    end
end

end

function similarityMatrix = cosineSimilarity(X, Y, ~)
n = size(X, 1);
c = size(Y, 1);
similarityMatrix = zeros(n, c);
for i = 1:n
    for j = 1:c
        dotProduct = dot(X(i, :), Y(j, :));
        normX = norm(X(i, :));
        normY = norm(Y(j, :));
        similarityMatrix(i, j) = dotProduct / (normX * normY);
    end
end
end

function S = AddLogicalLabel(S, Y, HyperPara)
for i = 1:HyperPara.ins_num
    y1 = Y(i,:);
    y2 = ~y1;
    w1 = sum(y1 == 1);
    w2 = sum(y1 == 0);
    if w1 ~= 0 && w2 ~=0
        S(i,:) = S(i,:) - y2*(1/(w2*size(Y,2)));
    end
end
end