function A = InstanceCorrelation(X, HyperPara)
A = InstanceSimilarity_PAR(X, HyperPara.k);
end

function A = InstanceSimilarity_PAR(X, k)
ins_num = size(X,1);
[~, neighbor] = pdist2(X, X, 'euclidean', 'Smallest', k+1);
neighbor = neighbor(2:end, :);
neighbor = neighbor';
rows = repmat((1:ins_num)', 1, k);
datas = zeros(ins_num, k);
for i=1:ins_num
    neighborIns = X(neighbor(i,:), :)';  % neighborIns: d*k
    w = lsqnonneg(neighborIns, X(i,:)');  % lsqnonneg:neighbor featMatrix, featVector\ w:weight
    datas(i,:) = w;
end
trans = sparse(rows, neighbor, datas, ins_num, ins_num);
sumW = full(sum(trans, 2));
sumW(sumW == 0) = 1;
trans = bsxfun(@rdivide, trans, sumW);
A = full(trans);
A = A + eye(size(A, 1));
end
