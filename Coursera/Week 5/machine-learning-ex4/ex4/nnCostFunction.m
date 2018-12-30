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
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

A1 = [ones(m, 1) X];
z2 = A1 * Theta1';
A2 = [ones(size(z2), 1) sigmoid(z2)];
z3 = A2 * Theta2';
A3 = sigmoid(z3);
H = A3;

yv = [1:num_labels] == y;
ttheta1 = Theta1(:,2:end);
ttheta2 = Theta2(:,2:end);
#fprintf('Size of X is [%d, %d]\n', size(X));
#fprintf('Size of A1 is [%d, %d]\n', size(A1));
#fprintf('Size of A2 is [%d, %d]\n', size(A2));
#fprintf('Size of A3 is [%d, %d]\n', size(A3));
#fprintf('Size of Yv is [%d, %d]\n', size(yv));
#fprintf('Size of ttheta1 is [%d, %d]\n', size(ttheta1));
#fprintf('Size of ttheta2 is [%d, %d]\n', size(ttheta2));

J1 = -1/m * sum(sum(yv .* log(H) + (1 - yv) .* log(1 - H)));
J2 = lambda / (2 * m) * (sum(sum(ttheta1 .* ttheta1)) + sum(sum(ttheta2 .* ttheta2)));

J = J1 + J2;

#fprintf('Size of J1 is [%d, %d]\n', size(J1));
#fprintf('Size of J2 is [%d, %d]\n', size(J2));

delta3 = A3 - yv; # [5000 x 10]
delta2 = delta3 * ttheta2 .* sigmoidGradient(z2); # [5000 x 25]
#fprintf('Size of delta3 is [%d, %d]\n', size(delta3));
#fprintf('Size of delta2 is [%d, %d]\n', size(delta2));

Theta2_grad = delta3' * A2; # [10 x 25] 
Theta1_grad = delta2' * A1; # [25 x 400]
#fprintf('Size of Theta2_grad is [%d, %d]\n', size(Theta2_grad));
#fprintf('Size of Theta1_grad is [%d, %d]\n', size(Theta1_grad));

Theta1_grad = Theta1_grad ./ m;
Theta2_grad = Theta2_grad ./ m;

Theta1_grad(:,2:end) = Theta1_grad(:,2:end) + lambda / m * Theta1(:,2:end);
Theta2_grad(:,2:end) = Theta2_grad(:,2:end) + lambda / m * Theta2(:,2:end);

% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
