function [W, A, D] = Optimization_PMSNE(X, Y, HyperPara)

%   (1/2)*(A_t*X*W-D, f)
% + alpha*tr(W'*B*W)
% + (1/2)*beta*(A_t*D*L-D, f)
% + (1/2)*gamma*(D-S, f)
% + (1/2)*delta*(Y_hat.*D, f)
% + (1/2)*(A_t*X-X, f)
% + epsilon*(A_t, 1)

[num_sample, num_dim]=size(X);
Y(Y == -1) = 0;
alpha = HyperPara.alpha;
beta  = HyperPara.beta;
gamma = HyperPara.gamma;
delta = HyperPara.delta;
epsilon = HyperPara.epsilon;
rho = 1e-4;
rhoI = 2;

maxIter = HyperPara.maxIter;
minLossMargin = HyperPara.minLossMargin;
closedform = HyperPara.closedform;
uselip = HyperPara.uselip;
disp('Optimization Start---');

%% Initialization for iteration
X = [X, ones(num_sample,1)];
W_t = (X'*X + rhoI*eye(num_dim+1))\(X'*Y);
W_t_1 = W_t;
A_t = InstanceCorrelation(X, HyperPara);
A_t_1 = A_t;
C_t = zeros(size(A_t, 1), size(A_t, 2));
M_t = zeros(size(A_t, 1), size(A_t, 2));
M_t_1 = M_t;
A_t = A_t - diag(diag(A_t));
S_t = PseudoDistribution(A_t*X, Y, Y, HyperPara);
L_t = LabelCorrelation(Y);
L_t = L_t - diag(diag(L_t));
D_t = S_t;
D_t_1 = D_t;

oldloss = Inf;
iter = 1;
Loss = zeros(1, maxIter);

%% Optimization in iter
lr = 1e-6;
while iter <= maxIter
    B_t = Proximate(W_t);
    if closedform == 0
        Grad_gA = GradientofgA(A_t, X, W_t, D_t, L_t, M_t, C_t, beta, rho);
        if uselip == 1
            A_t = A_t_1 - Grad_gA*(1/Lip_A);
        else
            stepsize_A = lr;
            A_t = A_t_1 - Grad_gA*stepsize_A;
        end
        A_t = A_t - diag(diag(A_t));
    elseif closedform == 1
        disp('error');
        return;
    end
    C_t = softthres(A_t + M_t_1, epsilon/rho);
    rho = min(rho*1.1, 1e10);
    M_t = M_t_1 + A_t - C_t;
    M_t_1 = M_t;
    Grad_D = GradientofD(A_t, D_t, L_t, X, W_t, S_t, Y, beta, gamma, delta);
    stepsize_D = lr;
    D_t = D_t_1 - Grad_D*stepsize_D;
    D_t_1 = D_t;
    if closedform == 0
        Grad_W = GradientofW(A_t, X, W_t, D_t, B_t, alpha);
        if uselip == 1
            W_t = W_t_1 - Grad_W*(1/Lip_W);
        else
            stepsize_W = lr;
            W_t = W_t_1 - Grad_W*stepsize_W;
        end
        W_t_1 = W_t;
    elseif closedform == 1
        disp('error');
        return;
    end
    % other updates
    if iter > 0 && iter < 100
        D_t = D_t .* Y;
        D_t_1 = D_t;
        S_t = PseudoDistribution_new(A_t*X, D_t, Y, HyperPara);
    end
    A_t = Constraint01(A_t);
    A_t_1 = A_t;
    % Calculate loss
    term_lse = 0.5*(norm(X*W_t-D_t, 'fro')^2);
    term_w21 = alpha*trace(W_t'*B_t*W_t);
    term_adl = 0.5*beta*(norm(A_t*D_t*L_t-D_t, 'fro')^2);
    term_ds  = 0.5*gamma*(norm(D_t-S_t, 'fro')^2);
    term_yd  = 0.5*delta*(norm((~Y).*D_t, 'fro')^2);
    term_ax  = 0.5*(norm(A_t*X-X, 'fro')^2);
    term_a1  = epsilon*norm(A_t, 1);
    currloss = term_lse + term_w21 + term_adl + term_ds + term_yd + term_ax + term_a1;
    Loss(iter) = currloss;
    if (abs(oldloss - currloss) < minLossMargin*abs(currloss)) && (iter > 3)
        break;
    end
    oldloss = currloss;
    iter = iter + 1;
end
W = W_t;
A = A_t;
D_t = D_t .* Y;
D = D_t;
% fine-tuning for W, cause the W is the weights of A*X, while input of MLKNN is X
iter = 1;
lambdaw = HyperPara.alpha;  % 0.1*size(X,1)/size(X,2);
while(1)
    I = diag(1./max(sqrt(sum((W).*(W),2)),eps));
    W = (X'*X+lambdaw*I)\(X'*D);
    if iter == 10
        break;
    end
    iter=1+iter;
end

end  % of Optimization

%% Calculate B
function B = Proximate(W)
num = size(W, 1);
B = zeros(num, num);
for i = 1:num
    temp = norm(W(i, :), 2);
    if temp ~= 0
        B(i, i) = 1/temp;
    else
        B(i, i) = 0;
    end
end
end

%% Gradient of W
function Grad_W = GradientofW(A_t, X, W, D, B, alpha)
Grad_W = (A_t*X)'*(A_t*X*W-D) + 2*alpha*B*W;
end

%% Gradient of A
function Grad_gA = GradientofgA(A, X, W, D, L, M, C, beta, rho)
lambda = 1;
Grad_gA = (A*X*W-D)*(W')*(X') + beta*(A*D*L-D)*(L')*(D') + lambda*(A*X-X)*(X') + rho*(A+M-C);
end

%% Gradient of D
function Grad_D = GradientofD(A_t, D, L, X, W, S, Y, beta, gamma, delta)
Y_hat = ~Y;
Grad_D = beta*A_t'*(A_t*D*L-D)*L' - (A_t*X*W-D) - beta*(A_t*D*L-D) + gamma*(D-S) + delta*(Y_hat.*D);
end

%% Soft Thresholding
function result = softthres(Matrix, lambda)
result = max(Matrix - lambda,0) - max(-Matrix - lambda,0);
end
