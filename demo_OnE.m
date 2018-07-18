rng('shuffle');
addpath('evaluation_tools');
data_dir = 'data/';

dataset = 'cifar10'; % 'sun397' 'cifar10',  'labelme' 
retain_dim = 512;
Ls = [8 16 24 32];

for L = Ls
    fprintf(2, 'Orthonromal Encoder - dataset: %s - #bits: %d\n', dataset, L);

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
    S = diag(S);

    gallery_features = U(:,1:min(retain_dim, d))' * gallery_features;
    test_features    = U(:,1:min(retain_dim, d))' * test_features;

    %% calculate the scale parameter and apply
    scale = sqrt(L/sum(S(1:L)));
    test_features    = test_features    * scale;
    gallery_features = gallery_features * scale;

    %% solve orthogonal projection matrix using OnE algorithm
    verbose = false;
    max_iter = 100;
    X = gallery_features';
    V = OnE(X, L, max_iter, verbose);
    % the processing time include pre-processing stage
    fprintf('Total learning time: %f\n', toc); 
    
    %% evaluate
    gallery_code = (gallery_features' * V > 0);
    test_code = (test_features' * V > 0);
    junk = [];
    map = KNNMap( gallery_code, test_code, nbase, gnd_inds, junk);
    fprintf('L: %d - mAP: %f\n', L, map);
end