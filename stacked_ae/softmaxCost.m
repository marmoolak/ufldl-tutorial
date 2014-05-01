function [cost, grad] = softmaxCost(theta, numClasses, inputSize, lambda, data, labels)

% numClasses - the number of classes (10)
% inputSize - the size N of the input vector (8)
% lambda - weight decay parameter
% data - the N x M input matrix, where each column data(:, i) corresponds to
%        a single test set (8 x 100)
% labels - an M x 1 matrix containing the labels corresponding for the input data
% (100 x 1)

% Unroll the parameters from theta
theta = reshape(theta, numClasses, inputSize); % 10 x 8
numCases = size(data, 2); % 100
groundTruth = full(sparse(labels, 1:numCases, 1)); % 10 x 100
cost = 0;
thetagrad = zeros(numClasses, inputSize); % 10 x 8

%% ---------- YOUR CODE HERE --------------------------------------
%  Instructions: Compute the cost and gradient for softmax regression.
%                You need to compute thetagrad and cost.
%                The groundTruth matrix might come in handy.
m = numCases;

M = theta * data; % 10 x 100
% minus the max value from each element to prevent overflow (see class notes)
M = bsxfun(@minus, M, max(M, [], 1));
expM = exp(M);
hypo = bsxfun(@rdivide, expM, sum(expM));
cost = -((sum(sum(groundTruth .* log(hypo)))) / m);

weightDecay = (lambda / 2) * sum(sum(theta .^ 2));
cost = cost + weightDecay;


thetagrad =  -((groundTruth - hypo) * data') / m;
thetagrad = thetagrad + lambda * theta;







% ------------------------------------------------------------------
% Unroll the gradient matrices into a vector for minFunc
grad = [thetagrad(:)];
end

