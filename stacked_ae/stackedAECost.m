function [ cost, grad ] = stackedAECost(theta, inputSize, hiddenSize, ...
                                              numClasses, netconfig, ...
                                              lambda, data, labels)
                                         
% stackedAECost: Takes a trained softmaxTheta and a training data set with labels,
% and returns cost and gradient using a stacked autoencoder model. Used for
% finetuning.
                                         
% theta: trained weights from the autoencoder
% visibleSize: the number of input units
% hiddenSize:  the number of hidden units *at the 2nd layer*
% numClasses:  the number of categories
% netconfig:   the network configuration of the stack
% lambda:      the weight regularization penalty
% data: Our matrix containing the training data as columns.  So, data(:,i) is the i-th training example. 
% labels: A vector containing labels, where labels(i) is the label for the
% i-th training example


%% Unroll softmaxTheta parameter

% We first extract the part which compute the softmax gradient
softmaxTheta = reshape(theta(1:hiddenSize*numClasses), numClasses, hiddenSize);

% Extract out the "stack"
stack = params2stack(theta(hiddenSize*numClasses+1:end), netconfig);

% You will need to compute the following gradients
softmaxThetaGrad = zeros(size(softmaxTheta));
stackgrad = cell(size(stack));
for d = 1:numel(stack)
    stackgrad{d}.w = zeros(size(stack{d}.w));
    stackgrad{d}.b = zeros(size(stack{d}.b));
end

cost = 0; % You need to compute this

% You might find these variables useful
m = size(data, 2);
groundTruth = full(sparse(labels, 1:m, 1));


%% --------------------------- YOUR CODE HERE -----------------------------
%  Instructions: Compute the cost function and gradient vector for 
%                the stacked autoencoder.
%
%                You are given a stack variable which is a cell-array of
%                the weights and biases for every layer. In particular, you
%                can refer to the weights of Layer d, using stack{d}.w and
%                the biases using stack{d}.b . To get the total number of
%                layers, you can use numel(stack).
%
%                The last layer of the network is connected to the softmax
%                classification layer, softmaxTheta.
%
%                You should compute the gradients for the softmaxTheta,
%                storing that in softmaxThetaGrad. Similarly, you should
%                compute the gradients for each layer in the stack, storing
%                the gradients in stackgrad{d}.w and stackgrad{d}.b
%                Note that the size of the matrices in stackgrad should
%                match exactly that of the size of the matrices in stack.
%


n = numel(stack);
z = cell(n+1, 1);
a = cell(n+1, 1);
a{1} = data;

for l=1:n
	z{l+1} = stack{l}.w * a{l} + repmat(stack{l}.b, 1, m);
	a{l+1} = sigmoid(z{l+1});
end

M = softmaxTheta * a{n+1};
M = bsxfun(@minus, M, max(M, [], 1));
expM = exp(M);
hypo = bsxfun(@rdivide, expM, sum(expM));
cost = -((sum(sum(groundTruth .* log(hypo)))) / m);

weightDecay = (lambda / 2) * sum(sum(softmaxTheta .^ 2));
cost = cost + weightDecay;

thetagrad =  -((groundTruth - hypo) * a{n+1}') / m;
softmaxThetaGrad = thetagrad + lambda * softmaxTheta;	

% now for the backprop gradient computation
delta = cell(n+1, 1);
delta{n+1} = -(softmaxTheta' * (groundTruth - hypo)) .* (a{n+1} .* (1 - a{n+1}));
for l=n:-1:2
	delta{l} = (stack{l}.w' * delta{l+1}) .* a{l} .* (1 - a{l});
end

for l=1:n
	stackgrad{l}.w = (delta{l+1} * a{l}') / m;
	stackgrad{l}.b = sum(delta{l+1}, 2) / m;
end



% Below is the non-generalized verion of the code, works only with a 2 layer autoencoder
% as used in the exercise (kept for readbility)

% W1 = stack{1}.w;
% b1 = stack{1}.b;
% W2 = stack{2}.w;
% b2 = stack{2}.b;

% 1: feedforward
% z2 = W1 * data + repmat(b1,1,m);
% a2 = sigmoid(z2);

% z3 = W2 * a2 + repmat(b2,1,m);
% a3 = sigmoid(z3);


% backprop: softMax first, then go backwards.
% M = softmaxTheta * a3; % 
% minus the max value from each element to prevent overflow (see class notes)
% M = bsxfun(@minus, M, max(M, [], 1));
% expM = exp(M);
% hypo = bsxfun(@rdivide, expM, sum(expM));
% cost = -((sum(sum(groundTruth .* log(hypo)))) / m);

% weightDecay = (lambda / 2) * sum(sum(softmaxTheta .^ 2));
% cost = cost + weightDecay;

% thetagrad =  -((groundTruth - hypo) * a3') / m;
% softmaxThetaGrad = thetagrad + lambda * softmaxTheta;	


% now for the backprop gradient computation
% delta3 = -(softmaxTheta' * (groundTruth - hypo)) .* (a3 .* (1 - a3));
% delta2 = (W2' * delta3) .* a2 .* (1 - a2);

% 4: calc gradient (do not regularize)
% W1grad = (delta2 * data') / m; % + lambda * W1;
% W2grad = (delta3 * a2')   / m; % + lambda * W2;
% b1grad = sum(delta2, 2) / m;
% b2grad = sum(delta3, 2) / m;

	
% stackgrad{1}.w = W1grad;
% stackgrad{1}.b = b1grad;
% stackgrad{2}.w = W2grad;
% stackgrad{2}.b = b2grad;




% -------------------------------------------------------------------------

%% Roll gradient vector
grad = [softmaxThetaGrad(:) ; stack2params(stackgrad)];

end


% You might find this useful
function sigm = sigmoid(x)
    sigm = 1 ./ (1 + exp(-x));
end
