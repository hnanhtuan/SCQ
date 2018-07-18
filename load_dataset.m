function [ gallery_features, gallery_labels, test_features, test_labels] = ...
                                            load_dataset(base_dir, dataset )
switch dataset
    case 'sun397'
        load([base_dir, 'SUN397/sun397-vggfc7_gallery.mat'], 'gallery_features', 'gallery_labels');
        load([base_dir, 'SUN397/sun397-vggfc7_test.mat'], 'test_features', 'test_labels');
        gallery_features = double(gallery_features');
        test_features = double(test_features');
        gallery_labels = gallery_labels - 1;
        test_labels = test_labels - 1;
    case 'cifar10'
        load([base_dir, 'cifar10/cifar10_vggfc7_gallery.mat'], 'gallery_features', 'gallery_labels');
        load([base_dir, 'cifar10/cifar10_vggfc7_test.mat'], 'test_features', 'test_labels');
    case 'labelme'
        load([base_dir,'labelme/labelme_vggfc7_gallery.mat'], 'gallery_features', 'gallery_labels');
        load([base_dir,'labelme/labelme_vggfc7_test.mat'], 'test_features', 'test_labels');
    otherwise
        error('Undefined dataset')
end
end

