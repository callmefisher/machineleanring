function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
TransTheta = theta';
TransX = X';
[M, FeatureCount] = size(X);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%
ThetaJParaCount = FeatureCount ;








%% 求Theta向量
%% Alpha = 0.01;
% for iter =1:15000
%     
%     tempTheta = TransTheta;
%     for thetaJ = 1:ThetaJParaCount      
%         FinalSum = 0;
%         for i = 1:m
%             TmpHx = 0;
%             for thetaInsideJ = 1:ThetaJParaCount
%                 TmpHx = TmpHx + tempTheta(1, thetaInsideJ) * TransX(thetaInsideJ, i);
%             end
%             gHx =  1 / (1 + exp(-TmpHx)) ;
%              FinalSum = FinalSum + (gHx - y(i)) * X(i, thetaJ) ;
%         end
%         DerivedNum = Alpha * FinalSum / m;
%         TransTheta( 1, thetaJ) = tempTheta( 1, thetaJ) -   DerivedNum;     
%     end
% end


% 求梯度
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
    DerivedNum = FinalSum / m;
    grad(thetaJ) = DerivedNum;
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

J = -FinalSum/m;




%% 向量化的方案
%J = (-1 / m) * (y' * log(sigmoid(X * theta)) + (1-y)' * log(1 - sigmoid(X*theta)));
%grad = (1 / m) * X' * (sigmoid(X * theta) - y);


%hθ(x) = g(z) = 1/1 + e^?z;
%J(θ) = -1/m( y(i) log(hθ(x(i))) + (1 ? y(i)) log(1 ? hθ(x(i))))
% θ= θ - a/m (hθ(x(i)) ? y(i))x(j)




% =============================================================

end
