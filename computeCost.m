function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
Transtheta = theta';
ThetaLen = length(Transtheta);
FinalSum = 0;
for i=1:m
   for j = 1:ThetaLen
    TmpInnerSum = 0;
    TmpInnerSum = TmpInnerSum + Transtheta(1,j)* X(j,i);
    TmpReduceNum = TmpInnerSum - y(m,1);
    FinalSum = FinalSum +  TmpReduceNum * TmpReduceNum;
   end
end

J = FinalSum / (2 * m);
% =========================================================================

end
