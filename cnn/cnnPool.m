function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(featureNum, imageNum, imageRow, imageCol)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(featureNum, imageNum, poolRow, poolCol)
%     

numImages = size(convolvedFeatures, 2);
numFeatures = size(convolvedFeatures, 1);
convolvedDim = size(convolvedFeatures, 3);

pooledFeatures = zeros(numFeatures, numImages, floor(convolvedDim / poolDim), floor(convolvedDim / poolDim));

% -------------------- YOUR CODE HERE --------------------
% Instructions:
%   Now pool the convolved features in regions of poolDim x poolDim,
%   to obtain the 
%   numFeatures x numImages x (convolvedDim/poolDim) x (convolvedDim/poolDim) 
%   matrix pooledFeatures, such that
%   pooledFeatures(featureNum, imageNum, poolRow, poolCol) is the 
%   value of the featureNum feature for the imageNum image pooled over the
%   corresponding (poolRow, poolCol) pooling region 
%   (see http://ufldl/wiki/index.php/Pooling )
%   
%   Use mean pooling here.
% -------------------- YOUR CODE HERE --------------------

for i = 1:numImages
    for j = 1:numFeatures
	for k = 1:floor(convolvedDim / poolDim)
	    for l = 1:floor(convolvedDim / poolDim)
		pooledFeature  = mean(mean(convolvedFeatures(j, i, ...
		    poolDim*(k-1)+1 : poolDim*(k), poolDim*(l-1)+1 : poolDim*(l))));
		pooledFeatures(j, i, k, l) = pooledFeature;
	    end
	end
    end
end


%  help figure this out
%  [1:poolDim x 1:poolDim], [1:poolDim x poolDim +1: poolDim *2]
%  [poolDim + 1: poolDim *2 x 1:poolDim], [poolDim + 1: poolDim*2 x poolDim + 1: poolDim*2]

%  pooldim*(k-1)+1 : poolDim*(k) x  pooldim*(l-1)+1 : poolDim*(l)



end

