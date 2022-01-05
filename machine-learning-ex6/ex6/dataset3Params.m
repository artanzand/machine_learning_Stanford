function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
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
% setting values to be tried for C and sigma
v = [0.01 0.03 0.1 0.3 1 3 10 30];
% keep answers for choices of C (row label) and sigma (column label)
Error = zeros(8,8);

for i = 1:length(v),     % C label (row)
  C_test = v(i);
  for j= 1:length(v),    % sigma label (column)
    sigma_test = v(j);
    % this goes inside loop to calculate the SVM model
    model = svmTrain(X, y, C_test, @(x1, x2) gaussianKernel(x1,x2,sigma_test));

    % using the model calculated above see what is the predictions vector
    predictions = svmPredict(model, Xval);
    
    % saving the response here
    Error(i,j) = mean(double(predictions ~= yval));
  endfor
  
endfor

% looking for lowest error value in the Answer matrix and giving the index
% row index gives optimized index of v(i) for and column index gives optimized v(j) for sigma

[m,ind] = min(Error(:));
[row,col] = ind2sub(size(Error),ind);

C = v(row);
sigma = v(col);
% =========================================================================

end
