function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

    % sum0 = 0;
    % sum1 = 0;
    % sum2 = 0;
    % for i = 1:m,
    %     sum0 += ((theta(1)*X(i, 1) + theta(2)*X(i, 2) + theta(3)*X(i, 3)) - y(i)) * X(i, 1);
    %     sum1 += ((theta(1)*X(i, 1) + theta(2)*X(i, 2) + theta(3)*X(i, 3)) - y(i)) * X(i, 2);
    %     sum2 += ((theta(1)*X(i, 1) + theta(2)*X(i, 2) + theta(3)*X(i, 3)) - y(i)) * X(i, 3);
    % end


    % tmp0 = theta(1) - alpha * ( (1/m) * sum0 ); 
    % tmp1 = theta(2) - alpha * ( (1/m) * sum1 );
    % tmp2 = theta(3) - alpha * ( (1/m) * sum2 );


    tmp0 = theta(1) - alpha * ( (1/m) * sum( ((X * theta )- y) .* X(:, 1) ) ); 
    for i =1:size(theta,1)
        theta(i) = theta(i) - alpha * ( (1/m) * sum( ((X * theta )- y) .* X(:, i) ) );
    end
    % ============================================================
    theta(1) = tmp0;
    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
