function [ V ] = OnE( X, L, max_iter, verbose )

if ~exist('verbose', 'var')
    verbose = false;
end

n = size(X, 1);     % # training samples
d = size(X, 2);     % feature dimension

% randomly initialize V
R = randn(d, d);
[Ur,~,~] = svd(R);
V = Ur(:,1:L);

% Compute dataset eigen values and eigen vectors
X_cov = (X'*X);
X_cov = (X_cov + X_cov')/2;
[U, S, ~ ] = svd(X_cov);
U = U(:, end:-1:1);
S = sort(diag(S));
Ssum = sum(S(:));

Z = X * V; 
B = -ones(size(Z));  
B(Z>=0) = 1;

prev_loss = 1e10;
for iter = 1:max_iter
    % fix B, update V
    V = solveOrthonormalV(B, X, U, S, V);
    
    % fix V, update B
    Z = X * V; 
    B = -ones(size(Z));  
    B(Z>=0) = 1;
    
    % compute loss
    current_loss = B - Z;
    loss = sum(current_loss(:).^2);
    step = abs(prev_loss - loss)/loss;
    prev_loss = loss; 
    
    % display
    if (verbose)
        var = svd((Z'*Z));
        total_var = sum(var(:));
        
        fprintf('Iter %d - loss: %.2f (step %f) - var %f (%.2f) \n', ...
                             iter, loss, step, total_var, 100*total_var/Ssum);
    end
end
end



