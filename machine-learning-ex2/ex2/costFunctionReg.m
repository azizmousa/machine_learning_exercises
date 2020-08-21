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
%              $
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


thetaTemp = theta;
thetaTemp(1) = 0;

thetasum = sum(thetaTemp.^2);
jreg = ( lambda / (2*m) ) * thetasum;
% jreg(1) = 0;
g = sigmoid(X * theta);
J = ((-1 /m) * sum( ((y) .* log(g)) + ( (1-y).*(log(1 - g)) ) )) + jreg;


error = g - y;
grad = ((1/m) * (error' * X) )' + (( lambda / m ) * thetaTemp);



% =============================================================

end
