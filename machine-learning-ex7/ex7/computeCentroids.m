function centroids = computeCentroids(X, idx, K)
%COMPUTECENTROIDS returns the new centroids by computing the means of the 
%data points assigned to each centroid.
%   centroids = COMPUTECENTROIDS(X, idx, K) returns the new centroids by 
%   computing the means of the data points assigned to each centroid. It is
%   given a dataset X where each row is a single data point, a vector
%   idx of centroid assignments (i.e. each entry in range [1..K]) for each
%   example, and K, the number of centroids. You should return a matrix
%   centroids, where each row of centroids is the mean of the data points
%   assigned to it.
%

% Useful variables
[m n] = size(X);

% You need to return the following variables correctly.
centroids = zeros(K, n);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every centroid and compute mean of all points that
%               belong to it. Concretely, the row vector centroids(i, :)
%               should contain the mean of the data points assigned to
%               centroid i.
%
% Note: You can use a for-loop over the centroids to compute this.
%

num_centroid = 1;
while num_centroid <= K
  sum_of_class = zeros(size(X(1,:)));
  total_in_class = 0;
  idx_index = 1;

  % Look through our list of indexes for -this- index
  while idx_index <= size(idx, 1)
    % When we find a match, add the corresponding vector from our data set onto
    % our running total
    if (idx(idx_index) == num_centroid)
      sum_of_class += X(idx_index, :);
      total_in_class += 1;
    end
    centroids(num_centroid, :) = (sum_of_class / total_in_class);
    idx_index += 1;
  end
  num_centroid += 1;
end






% =============================================================


end

