clear all;
clc;

%% 获取一批线性不可分的数据
center = [3, 4];
num_point = [130, 150];
round = 2;
ring = [2, 3.5];

round_data = get_round_data(center(1), center(2), round, num_point(1));
label_round = zeros(size(round_data, 1), 1);
ring_data = get_ring_data(center(1), center(2), ring(2), ring(1), num_point(2));
label_ring = ones(size(ring_data, 1), 1);

hold on;
scatter(round_data(:, 1), round_data(:, 2), 'r^');
scatter(ring_data(:, 1), ring_data(:, 2), 'bx');
title('SVM非线性分类展示');


data = cat(1, round_data, ring_data);
label = cat(1, label_round, label_ring);

random_index = randperm(size(data, 1))';

train_data = data(random_index(1:floor(0.7 * size(data, 1))), :);
train_label = label(random_index(1:floor(0.7 * size(data, 1)), :));

test_data = data(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1)), :);
test_label = label(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1), :));

scatter(test_data(:, 1), test_data(:, 2), 80, 'ko');

%% 训练C-SVC模型
start = cputime;
model = svmtrain(train_label, train_data, '-c 1 -g 0.07');  %训练模型，使用高斯核函数
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_data, model);  %用模型预测
fprintf('training time:%.3f\n', cputime - start);

%% 结果绘图展示
num_grid = 300;
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
