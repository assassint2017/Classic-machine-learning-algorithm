% 获取一批线性不可分数据的脚本

clear all;
clc;

center = [3, 4];
num_point = [130, 150];
round = 2;
ring = [2, 3.5];

round_data = get_round_data(center(1), center(2), round, num_point(1));
label_round = zeros(size(round_data, 1), 1);
ring_data = get_ring_data(center(1), center(2), ring(2), ring(1), num_point(2));
label_ring = zeros(size(ring_data, 1), 1);

% 为了体现特征尺度统一的重要性，下面将其中一个维度上的数据扩大倍，在同样的参数下，来观察分类效果
% round_data(:, 1) = round_data(:, 1) * 10;
% ring_data(:, 1) = ring_data(:, 1) * 10;

subplot(121);
hold on;
scatter(round_data(:, 1), round_data(:, 2), 'r^');
scatter(ring_data(:, 1), ring_data(:, 2), 'bx');
title('原始数据');
legend('负样本', '正样本')
hold off;

data = cat(1, round_data, ring_data);
label = cat(1, label_round, label_ring);

random_index = randperm(size(data, 1))';

train_data = data(random_index(1:floor(0.7 * size(data, 1))), :);
train_label = label(random_index(1:floor(0.7 * size(data, 1)), :));

test_data = data(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1)), :);
test_label = label(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1), :));

subplot(122);
hold on;
scatter(train_data(:, 1), train_data(:, 2), 'g');
scatter(test_data(:, 1), test_data(:, 2), 'y');
title('训练数据');
legend('训练数据', '测试数据');
hold off;

start = cputime;
model = svmtrain(train_label, train_data, '-c 1 -g 0.07');  %训练模型
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_data, model);  %用模型预测
fprintf('training time:%.3f\n', cputime - start);
