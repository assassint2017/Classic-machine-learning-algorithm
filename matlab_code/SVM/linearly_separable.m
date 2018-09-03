% ��ȡһ�����Կɷ����ݵĽű�
% ���Խ��������ĵ�ľ������õ���Խ�һЩ���������ݻ������Բ��ɷֵģ�������Ȼ����ʹ�ô���
% ����������SVM���з���

clear all;
clc;

%% ��ȡһ�����Կɷֵ�����
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
title('SVM���Է���չʾ');

data = cat(1, data_1, data_2);
label = cat(1, label_1, label_2);

random_index = randperm(size(data, 1))';

train_data = data(random_index(1:floor(0.7 * size(data, 1))), :);
train_label = label(random_index(1:floor(0.7 * size(data, 1)), :));

test_data = data(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1)), :);
test_label = label(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1), :));

scatter(test_data(:, 1), test_data(:, 2), 80, 'ko');

%% ѵ��C-SVCģ��
start = cputime;
model = svmtrain(train_label, train_data, '-c 1 -t 0');  %ѵ��ģ�ͣ�ʹ�����Ժ˺���
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_data, model);  %��ģ��Ԥ��
fprintf('training time:%.3f\n', cputime - start);

%% �����ͼչʾ
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
legend('������', '������', '��������', '֧������', '�½�', '���߽߱�', '�Ͻ�');
hold off;