rng('shuffle');
addpath('evaluation_tools');
data_dir = 'data/';

dataset = 'cifar10';  % 'cifar10', 'sun397', 'labelme'
retain_dim = 512;
lambda = 0.02;
Ls = [8 16 24 32];

for L = Ls
    fprintf(2, 'Orthogonal Encoder - dataset: %s - #bits: %d - lambda: %0.02f\n', ...
                                dataset, L, lambda);
    
    %% load dataset and compute the groundtruth for evaluation
    [ gallery_features, gallery_labels, test_features, test_labels] = load_dataset(data_dir, dataset ); 
    d = size(gallery_features, 2);
    nquery = length(test_labels);
    nbase = length(gallery_labels);
    gnd_inds = zeros(nbase/10, nquery);
    for label = min(test_labels(:)):max(test_labels(:))
        gnd = find(gallery_labels == label);
        idx = find(test_labels == label);
        gnd_inds(:, idx) = gnd*ones(1,length(idx));
    end
    gnd_inds = gnd_inds';

    %% Pre-processing: 
    % + zero-mean data
    % + reduce dimensionality for computational efficiency
    tic
    gallery_features = gallery_features'; 
    test_features    = test_features';

    desc_mean        = mean (gallery_features, 2);
    gallery_features = bsxfun (@minus, gallery_features, desc_mean);
    test_features    = bsxfun (@minus, test_features, desc_mean);
    Xcov = gallery_features * gallery_features';
    Xcov = (Xcov + Xcov') / (2 * size (gallery_features, 2));    % make it more robust
    [U, S, ~] = svd( Xcov );
    gallery_features = U(:,1:min(retain_dim, d))' * gallery_features;            % PCA
    test_features    = U(:,1:min(retain_dim, d))' * test_features;               % PCA
    
    %% calculate the scale parameter and apply
    scale = sqrt(L/sum(S(1:L)));
    test_features    = test_features    * scale;
    gallery_features = gallery_features * scale;

	%% solve orthogonal projection matrix using OnE algorithm
    verbose = true;
    max_iter = 50;
    X = gallery_features';
    V = OgE(X, L, lambda, max_iter, verbose);
    % the processing time include pre-processing stage
    fprintf('Total learning time: %f\n', toc);

    %% evaluate
    gallery_code = (gallery_features' * V > 0);
    test_code = (test_features' * V > 0);
    junk = [];
    map = KNNMap( gallery_code, test_code, nbase, gnd_inds, junk);
    fprintf('L: %d - mAP: %f\n', L, map);

end