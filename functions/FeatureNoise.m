function X_noisy = FeatureNoise(X, lambda)
if lambda == 0
    X_noisy = X;
else
    [U, W, T] = svd(X);
    rankX = nnz(W);
    k = round(lambda*rankX);  % number of noisy bases
    orthX = U(:, rankX+1:end);  % orth space
    index = randi(size(orthX, 2), 1, k);
    A = orthX(:, index);  % selected matrix
    U1 = U;
    U1(:, rankX-k+1:rankX) = A;
    X_noisy = U1*W*T;
end
end