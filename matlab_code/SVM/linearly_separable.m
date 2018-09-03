% 获取一批线性可分数据的脚本
% 可以将两个中心点的距离设置的相对近一些，这样数据会变成线性不可分的，但是依然可以使用带有
% 软间隔的线性SVM进行分类

clear all;
clc;

%% 获取一批线性可分的数据
center_1 = [3, 4];
center_2 = [5, 6];
num_point = [130, 150];
sparse_degree = [0.7, 1.0];

data_1 = get_gauss_data(center_1(1), center_1(2), sparse_degree(1), num_point(1));
label_1 = zeros(size(data_1, 1), 1);
data_2 = get_gauss_data(center_2(1), center_2(2), sparse_degree(2), num_point(2));
label_2 = ones(size(data_2, 1), 1);

hold on;
scatter(data_1(:, 1), data_1(:, 2), 'r^');
scatter(data_2(:, 1), data_2(:, 2), 'bx');
title('SVM线性分类展示');

data = cat(1, data_1, data_2);
label = cat(1, label_1, label_2);

random_index = randperm(size(data, 1))';

train_data = data(random_index(1:floor(0.7 * size(data, 1))), :);
train_label = label(random_index(1:floor(0.7 * size(data, 1)), :));

test_data = data(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1)), :);
test_label = label(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1), :));

scatter(test_data(:, 1), test_data(:, 2), 80, 'ko');

%% 训练C-SVC模型
start = cputime;
model = svmtrain(train_label, train_data, '-c 1 -t 0');  %训练模型，使用线性核函数
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_data, model);  %用模型预测
fprintf('training time:%.3f\n', cputime - start);

%% 结果绘图展示
num_grid = 100;
total_data = cat(1, test_data, train_data);
min_x = min(total_data(:, 1));
max_x = max(total_data(:, 1));
min_y = min(total_data(:, 2));
max_y = max(total_data(:, 2));
gridx = linspace(min_x, max_x, num_grid);
gridy = linspace(min_y, max_y, num_grid);
[gridX, gridY] = meshgrid(gridx, gridy);
grid_data = cat(2, gridX(:), gridY(:));
[predict_label, accuracy, dec_values] = svmpredict(zeros(length(grid_data), 1), grid_data, model);
scatter(model.SVs(:, 1), model.SVs(:, 2), 80, 'go')
contour(gridX, gridY, reshape(dec_values, num_grid, num_grid), [-1,-1]);
contour(gridX, gridY, reshape(dec_values, num_grid, num_grid), [0,0]);
contour(gridX, gridY, reshape(dec_values, num_grid, num_grid), [1,1]);
legend('正样本', '负样本', '测试数据', '支持向量', '下界', '决策边界', '上界');
hold off;