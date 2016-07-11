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

predictions = sigmoid(X * theta);                                               
errors = -y.*log(predictions) - (1.-y).*log(1 .- predictions);                               
J = 1/m * sum(errors) + (lambda / (2*m)) * sum(theta(2:end, 1) .^ 2);                                                          
                                                                                
for i = 1:size(X,2)                                                             
        gradErrors = (predictions - y) .* X(:,i);
	if (i == 1)
		grad(i,1) = 1/m * sum(gradErrors);
	else                               
        	grad(i,1) = 1/m * sum(gradErrors) + (lambda / m) * theta(i, 1);
	endif                                      
                                              




% =============================================================

end
