function X = SimplexProj(Y)  % proj_1
% X = Proj1(Y);
X = Proj1(Y);
X(isnan(X)) = 0;
end

function X = Proj1(Y)
[N,D] = size(Y);
X = sort(Y,2,'descend');
Xtmp = (cumsum(X,2)-1)*diag(sparse(1./(1:D)));
X = max(bsxfun(@minus,Y,Xtmp(sub2ind([N,D],(1:N)',sum(X>Xtmp,2)))),0);
end

function Y = Proj2(Y)
[N,~] = size(Y);
for i = 1:N
    d = Y(i,:);
    mind = min(d);
    di = d - mind;
    summindi = sum(di);
    dproj = di./summindi;
    Y(i,:) = dproj;
end
end