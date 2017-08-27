function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = ones(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

minima = zeros(size(X,1), 1);

% Initialize minima: set its values to the distance from the first centroid
% (consistent with having set idx equal to 1 in all cases).
i_x = 1;
for datum = X'
  minima(i_x) = norm(datum - centroids'(:, 1));
  i_x += 1;
end

% See if we can do better than centroid #1.
i_x = 1;
for datum = X'
  i_c = 2;
  while i_c <= K
    distance = norm(datum - centroids'(:, i_c));
    if (distance < minima(i_x))
      minima(i_x) = distance;
      idx(i_x) = i_c;
    end
    i_c += 1;
  end
  i_x += 1;
end

% compute distance from X to centroid
% if lower than current distance, store in idx






% =============================================================

end

