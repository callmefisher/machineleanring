function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);   % Theta的维度 = 分类器的个数 * (特征数量 + 1)

% Add ones to the X data matrix
X = [ones(m, 1) X];
Classfiers = num_labels;
% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%

TransX = X';
Alpha = 0.01;
for i = 1:Classfiers

   % TmpRightOutputY = y(i, :);
    TmpTransTheta0 = all_theta(i, :);
    %TmpTransTheta0 = TmpSaveTheta';

    ThetaJParaCount = size(TmpTransTheta0, 2);
    
    %% 迭代一定次数,求出
    for iter = 1:500
        
        TmpTransTheta1 = TmpTransTheta0;
        for thetaJ = 1:ThetaJParaCount
            FinalSum = 0;
            for example = 1:m
                TmpHx = 0;
                for thetaInsideJ = 1:ThetaJParaCount
                    TmpHx = TmpHx + TmpTransTheta1(1, thetaInsideJ) * TransX(thetaInsideJ, example);
                end
                gHx =  1 / (1 + exp(-TmpHx)) ;
                FinalSum = FinalSum + (gHx - y(example, 1 )) * X(example, thetaJ) ;
            end
            
            if thetaJ == 1 
                TmpTransTheta0(1, thetaJ) = TmpTransTheta1(1, thetaJ) - Alpha * FinalSum / m;
            else
                TmpTransTheta0(1, thetaJ) = TmpTransTheta1(1, thetaJ) - Alpha * FinalSum / m - lambda * TmpTransTheta1(1, thetaJ) / m;
            end    
        end  
    end
    
     all_theta(i, :) = TmpTransTheta0';
    
     

    
end



 %     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%
% 
% for i = 1:num_labels
%     [theta] = ...
%         fmincg(@(t)(lrCostFunction(t,X,(y == i),lambda)), ...
%             initial_theta, options);
%     all_theta(i,:) = theta';
% end







% =========================================================================


end
