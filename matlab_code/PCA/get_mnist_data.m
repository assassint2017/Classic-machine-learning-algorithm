function [train_data, train_label, test_data, test_label] = get_mnist_data()
% 读取本地的MNIST数据集中的数据
% 对于MNIST的图片来说，分辨率为28*28 共784维输入特征

train_dir = 'F:\matlab_code\data\mnist_img\train\';
test_dir = 'F:\matlab_code\data\mnist_img\test\';

folder(1,1) = {train_dir};
folder(1,2) = {test_dir};

for folder_index = 1:length(folder)
    subfolders_dir = dir(folder{folder_index});
    subfolders = {subfolders_dir.name}';
    subfolders = subfolders(3:length(subfolders));

    data = cell(10, 1);
    for i = 1:10
        imgs = dir(fullfile(folder{folder_index}, subfolders{i}, '*.png'));
        img_dir = {imgs.name}';
        for j = 1:length(img_dir)
            img = imread(fullfile(folder{folder_index}, subfolders{i}, img_dir{j}));
            grayimg = rgb2gray(img);
            if j == 1
                temp_data = grayimg(:)';
            else
                temp_data = cat(1, grayimg(:)', temp_data);
            end
        end
        data(i, 1) = {temp_data};
    end

    for i = 1:10
        if i == 1
            label = ones(length(data{i}), 1) * i - 1;
            final_data = data{i};
        else
            label = cat(1, ones(length(data{i}), 1) * i - 1, label);
            final_data = cat(1, data{i}, final_data);
        end
    end
    
    % 随机打乱数据
    random_index = randperm(length(final_data));
    final_data = final_data(random_index, :);
    label = label(random_index, :);

    % 由于SVM的输入数据类型有要求，所以需要进行转换
    final_data = double(final_data);
    label = double(label);
    
    if folder_index == 1
        train_data = final_data;
        train_label = label;
    else
        test_data = final_data;
        test_label = label;
    end
end

end