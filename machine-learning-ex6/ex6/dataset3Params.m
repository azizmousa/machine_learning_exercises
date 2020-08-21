function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%


pram_choise = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
n = length(pram_choise);
prams_eval = zeros(n*n, 3);
for i = 1:n,
	for j = 1:n,
		model = svmTrain(X, y, pram_choise(i), @(x1, x2) gaussianKernel(x1, x2, pram_choise(j)));
		predictions = svmPredict(model, Xval);
		prediction_error = mean(double(predictions ~= yval));
		prams_eval(j + (n*(i-1)),:) = [pram_choise(i), pram_choise(j), prediction_error]; 
	end
end

[val minimum_error] = min(prams_eval(:, 3));

C = prams_eval(minimum_error(1), 1);
sigma = prams_eval(minimum_error(1), 2);;
% =========================================================================

end
