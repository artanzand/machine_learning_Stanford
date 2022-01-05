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
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% for the cost function we don't want to count theta(1). We first construct a zeta matrix w/o it
zeta = theta(2:size(theta));

% calculating regularized J
J = 1/m * ( -y' * log(sigmoid(X * theta)) - (1 - y)' * log(1 - sigmoid(X * theta)) ) + lambda/(2*m) * sum (zeta .^2);

% update grad(1) by multiplying with ones(1,m) is the same as summing all elements
grad(1) = 1/m * sum(sigmoid(X * theta) - y);

% update the rest of grad from 2 to n+1
for i=2:size(theta),
  grad(i) = 1/m * X(:,i)' * (sigmoid(X * theta) - y) + lambda/m * theta(i);
endfor

% =============================================================

end
