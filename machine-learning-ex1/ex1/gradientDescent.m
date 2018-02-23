function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

NormalX = X(:,2);
X1 = (NormalX - mean(NormalX)) ./ ( max(NormalX) - min(NormalX));
y = (y - mean(y)) ./ (max(y) - min(y));
X = [ones(m, 1), X1];


Transtheta = theta';  %% 1 * N
ThetaLines = length(Transtheta);
TransDesignX = X';          %% N * M  (N 是特征数目 M是训练样本个数)

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    FinalSum = 0;
    for j = 1:ThetaLines
       
        for i = 1:m
            TmpInnerSum = 0;
            for featureCount = 1:ThetaLines
                TmpInnerSum = TmpInnerSum + Transtheta(1, featureCount) * TransDesignX(featureCount, i);
            end
             FinalSum = FinalSum + (TmpInnerSum - y(i,1)) * X(i, j) ; 
        end
        Transtheta( 1, j) = Transtheta( 1, j) -  alpha * FinalSum / m;
    end
    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

  theta = Transtheta';

end
