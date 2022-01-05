function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Adding column of ones to X
a1 = [ones(m,1) X];

% Creating Y in shape of matrix of vectors [0 0 1 0 ...] for each y(i)
Y = zeros(m, num_labels);
for i=1:m,
  Y(i, y(i)) = 1;
endfor

% Calculating unregularized cost function by going over examples
for i=1:m,
  % Forward propagation
  z2 = Theta1 * a1(i,:)';
  a2_temp = sigmoid(z2);
  a2 = [1; a2_temp];

  z3 = (Theta2 * a2);
  a3 = sigmoid(z3);     % this is h(x)
  
  J_temp = 1/m * ( -Y(i,:) * log(a3) - (1 - Y(i,:)) * log(1 - a3)) ;
  J = J + J_temp; 
endfor

% Calculating regularized cost function - theta(num_labels:end) will ignore the first column
J = J + lambda/(2*m) * sum(Theta1((hidden_layer_size + 1):end).^2) + ...
        lambda/(2*m) * sum(Theta2((num_labels + 1):end).^2);

% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
Delta1 = zeros(size(Theta1));
Delta2 = zeros(size(Theta2));

for i=1:m,
  % Forward propagation
  z2 = Theta1 * a1(i,:)';
  a2 = [1; sigmoid(z2)];

  z3 = (Theta2 * a2);
  a3 = sigmoid(z3);     % this is h(x)
  
  %Calculating deltas   % should add 1 to z2 ?
  delta3 = a3 - Y(i,:)';
  delta2 = (Theta2' * delta3) .* [1; sigmoidGradient(z2)];
  delta2 = delta2(2:end);   % removing the bias
  
  Delta2 = Delta2 + delta3 * a2';
  Delta1 = Delta1 + delta2 * a1(i,:);  
endfor

Theta1_grad = (1/m) * Delta1;
Theta2_grad = (1/m) * Delta2;

% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% since the first column of theta corresponds to bias units, I first calculate
% the addition regularized matrix and the set the first column to zeros

Reg1 = (lambda/m) * Theta1;
Reg1(:,1) = 0;
Theta1_grad = Theta1_grad + Reg1;

Reg2 = (lambda/m) * Theta2;
Reg2(:,1) = 0;
Theta2_grad = Theta2_grad + Reg2;

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
