function centroids = kMeansInitCentroids(X, K)
%KMEANSINITCENTROIDS This function initializes K centroids that are to be 
%used in K-Means on the dataset X
%   centroids = KMEANSINITCENTROIDS(X, K) returns K initial centroids to be
%   used with the K-Means on the dataset X
%

% You should return this values correctly
centroids = zeros(K, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should set centroids to randomly chosen examples from
%               the dataset X
%

m = size(X, 1);
idx = zeros(K, 1);

% randidx = randperm(size(X, 1));
% centroids = X(randidx(1:K), :);

for i = 1:K
    new_idx = randi(m, 1, 1);
    while new_idx == 0 || any(idx == new_idx) > 0
        new_idx = randi(m, 1, 1);
    endwhile
    idx(i) = new_idx;
endfor

centroids = X(idx,:);

% =============================================================

end

