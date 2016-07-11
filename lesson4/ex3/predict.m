function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% strategy: input is a bunch of examples, with each example being a row in X. 
% For each one of these examples, we need to use the neural network to predict
% the correct class. We can call the neural network some function h that takes
% an example xi and gives you back the class. But how does this work? The neural 
% network uses already trained network parameters theta1 and theta2, so you don't 
% have to worry about actually getting the predictions. But h does give back a
% vector that has various index which map to the num_label, probability. So that 
% is where the max comes in.

% Add ones to the X data matrix                                                 
X = [ones(m, 1) X]; 
% Get the hidden layer
% Theta1 is 25x401. The example x_i should be 401x1. But X is somethingx401.
% So we need to take the transpose of X to have the desired effect. If X were
% simply 1 example (1x401), we would get a 25x1 vector. But now we get a 
% 25xsomething vector. So hidden is 25xsomething
hidden = Theta1 * X';
% Now we add 1 as the top row of hidden, so hidden is 26xsomething

hiddenSig = sigmoid(hidden)
hiddenSig = [ones(1, m); hiddenSig;];

% Theta2 is 10x26, hiddenSig is 26xsomething, so hidden2 is 10xsomething
hidden2 = Theta2 * hiddenSig;
finalOutput = sigmoid(hidden2);

% something = 1 if X only has 1 example. Each column in something corresponds
% to an extra example in X. So we need to find the max index of each column,
% stick it in p;
[maxes, indices] = max(finalOutput);
p = indices';
% =========================================================================


end
