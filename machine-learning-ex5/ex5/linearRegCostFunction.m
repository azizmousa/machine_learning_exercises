function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

% j(theta) = (1/2m) * sum( (hyp(x) - y )^2 ) + (lambda/2m)*sum(theta^2)

theta0 = theta;
theta0(1) = 0;
hypothesis = X * theta;
reg =  ((lambda/(2.0*m)) * sum((theta0).^2));

J = ((1/(2*m)) * sum((hypothesis - y).^2)) + reg;



% grad = (1/m) * sum( error' * X(i)) + (lambda/m)*theta(i)
error = (hypothesis - y);
regularization = (lambda/m) * theta;
regularization(1) = 0;
grad = (1/m) * (error' * X);
grad += regularization';

% =========================================================================

grad = grad(:);

end
