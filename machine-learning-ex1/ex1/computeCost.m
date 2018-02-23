function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y
% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.
Transtheta = theta';
ThetaLen = length(Transtheta);
FinalSum = 0;
DesignX = X';
%%[XColume, XLine] = size(DesignX);
%%[ThetaColm, ThetaLine] = size(Transtheta);
%%fprintf("特征矩阵X (Colume:%d  Line:%d) 样本数量:%d Theta矩阵(Colume:%d  Line:%d)\n", XColume, XLine, m, ThetaColm, ThetaLine);
for i=1:m
   TmpInnerSum = 0;
   for j = 1:ThetaLen
    TmpInnerSum = TmpInnerSum + Transtheta(1,j)* DesignX(j,i);
   end
    TmpReduceNum = TmpInnerSum - y(i,1);
    FinalSum = FinalSum +  TmpReduceNum * TmpReduceNum;
end

J = FinalSum / (2 * m);
% =========================================================================

end
