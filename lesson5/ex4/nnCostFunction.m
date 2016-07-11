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
n = size(y, 1);
         
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

X = [ones(m, 1) X]; 
hidden = Theta1 * X';
hiddenSig = sigmoid(hidden);
hiddenSiga = [ones(1, m); hiddenSig;];
hidden2 = Theta2 * hiddenSiga;
finalOutput = sigmoid(hidden2);

% set up the y matrix
emptyZeros = zeros(num_labels, n);
for i = 1:n
	emptyZeros(:, i) = zeros(num_labels, 1);
	emptyZeros(y(i), i) = 1;
endfor

% backprop
d1 = 0;
d2 = 0;                                                                      
delta3 = finalOutput - emptyZeros;
%printf("Size of Theta1: %d x %d\n", size(Theta1, 1), size(Theta1, 2));
%printf("Size of Theta2: %d x %d\n", size(Theta2, 1), size(Theta2, 2));
%printf("Size of delta3: %d x %d\n", size(delta3, 1), size(delta3, 2)); 
delta2 = (Theta2' * delta3) .* sigmoidGradient([ones(1, m); hidden;]);
%printf("%s\n", mat2str(delta2));
delta2 = delta2(2:end, :);
%printf("Size of delta2: %d x %d\n", size(delta2, 1), size(delta2, 2)); 
%printf("Size of X: %d x %d\n", size(X, 1), size(X, 2));       
d2 = delta3 * (hiddenSiga');
d1 = delta2 * X;
reg1_grad = (lambda / m) * Theta1;
reg1_grad(:, 1) = zeros(size(Theta1, 1), 1);
reg2_grad = (lambda / m) * Theta2;                                              
reg2_grad(:, 1) = zeros(size(Theta2, 1), 1);   
Theta1_grad = d1 / m + reg1_grad;
Theta2_grad = d2 / m + reg2_grad;
%printf("Size of 1: %d x %d\n", size(Theta1_grad, 1), size(Theta1_grad, 2)); 
%printf("Size of 2: %d x %d\n", size(Theta2_grad, 1), size(Theta2_grad, 2)); 
newMatrix = -emptyZeros .* log(finalOutput) - (1 .- emptyZeros) .* log(1 .- finalOutput);
J = sum(newMatrix(:)) / n + (lambda / (2 * m)) * (sum((Theta1(:, 2:size(Theta1, 2)) .* Theta1(:, 2:size(Theta1, 2)))(:)) + sum((Theta2(:, 2:size(Theta2, 2)) .* Theta2(:, 2:size(Theta2, 2)))(:)));
% printf("%s\n", mat2str(y)); 










% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
