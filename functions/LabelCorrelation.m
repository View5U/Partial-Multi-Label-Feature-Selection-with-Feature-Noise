function L = LabelCorrelation(Y)
[~, c] = size(Y);
L = zeros(c, c);
for i = 1:c
    N = sum(Y(:, i)==1);
    for j = 1:c
        L(i, j) = sum(Y(:, i)'*Y(:, j))/N;
    end
end
L(isnan(L)) = 0;
end
