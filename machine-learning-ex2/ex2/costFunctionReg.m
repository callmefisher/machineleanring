function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

grad = zeros(size(theta));
TransTheta = theta';
TransX = X';
ThetaJParaCount = size(X, 2);

%计算lamta/2m * (ThetaJ^2)


% 求带lamda参数的梯度
for thetaJ = 1:ThetaJParaCount
    FinalSum = 0;
    for i = 1:m
        TmpHx = 0;
        for thetaInsideJ = 1:ThetaJParaCount
            TmpHx = TmpHx + TransTheta(1, thetaInsideJ) * TransX(thetaInsideJ, i);
        end
        gHx =  1 / (1 + exp(-TmpHx)) ;
        FinalSum = FinalSum + (gHx - y(i)) * X(i, thetaJ) ;
    end
    
    if thetaJ == 1
        grad(thetaJ) = FinalSum / m ;
    else
        grad(thetaJ) = FinalSum / m + lambda * theta(thetaJ) / m;
    end
    
    
end

FinalThetaJ = 0;
for j = 2:ThetaJParaCount
    FinalThetaJ = FinalThetaJ + theta(j) * theta(j);
end

%% 求Cost Function
FinalSum = 0;
for i = 1:m
    TmpHx = 0;
    for thetaInsideJ = 1:ThetaJParaCount
        TmpHx = TmpHx + TransTheta(1, thetaInsideJ) * TransX(thetaInsideJ, i);
    end
    gHx =  1 / (1 + exp(-TmpHx));
    LogHx =  y(i) * log(gHx) + (1 - y(i)) * log(1 - gHx);
    FinalSum = FinalSum + LogHx ;
end




J = -FinalSum/m + lambda * FinalThetaJ/ (2 * m);


% =============================================================

end
