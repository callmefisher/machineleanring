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
Alpha = 0.01;
for iter =1:1500
    
    tempTheta = TransTheta;
    for thetaJ = 1:ThetaJParaCount      

        for i = 1:m
            TmpInnerSum1 = 0;
            for thetaInsideJ = 1:ThetaJParaCount
                TmpInnerSum0 =  tempTheta(1, thetaInsideJ) * TransX(thetaInsideJ, i);
            end
             TmpInnerSum1 = TmpInnerSum1 + (log2(1 / (1 + exp(-TmpInnerSum0)) ) - y(i)) * X(i, thetaInsideJ) ;
        end
        TransTheta( 1, thetaJ) = tempTheta( 1, thetaJ) -  Alpha * TmpInnerSum1 / m;
    end 
end


TotalJ1 = 0;
for i = 1:m
    TmpInnerSum1 = 0;
   for featureJ = 1:FeatureCount
       TmpInnerSum0 =  TmpInnerSum0 + TransTheta(1, featureJ) * TransX(featureJ, i);
       TmpInnerSum1 = 1 / (1 + exp(-TmpInnerSum0));
   end
   TotalJ1 = TotalJ1 + log2(TmpInnerSum1) * y(i) + (1 - y(i)) * log2(1- TmpInnerSum1);
end
J = -TotalJ1 / m;

%h¦È(x) = g(z) = 1/1 + e^?z;
%J(¦È) = -1/m( y(i) log(h¦È(x(i))) + (1 ? y(i)) log(1 ? h¦È(x(i))))
% ¦È= ¦È - a/m (h¦È(x(i)) ? y(i))x(j)




% =============================================================

end
