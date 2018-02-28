function plotData(X, y)
%PLOTDATA Plots the data points X and y into a new figure 
%   PLOTDATA(x,y) plots the data points with + for the positive examples
%   and o for the negative examples. X is assumed to be a Mx2 matrix.

% Create New Figure
figure; hold on;

% ====================== YOUR CODE HERE ======================
% Instructions: Plot the positive and negative examples on a
%               2D plot, using the option 'k+' for the positive
%               examples and 'ko' for the negative examples.
%
Positive = find(y == 1);
Negtive = find(y == 0);
%plot(X(Positive, 1), X(Positive, 2), 'k+','LineWidth', 2, 'MarkerSize', 7);
plot(X(Positive, 1), X(Positive, 2) , '+', 'MarkerSize', 4);
plot(X(Negtive,1), X(Negtive,2), 'o', 'MarkerSize', 4);
% xlabel('A项考试分数');
% ylabel('B项考试分数');
% title('样本分布');
% legend('A样本', 'B样本', 'Location', 'northeast');


% =========================================================================



hold off;

end
