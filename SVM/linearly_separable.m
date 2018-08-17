% ��ȡһ�����Կɷ����ݵĽű�
% ���Խ��������ĵ�ľ������õ���Խ�һЩ���������ݻ������Բ��ɷֵģ�������Ȼ����ʹ�ô���
% ����������SVM���з���

clear all;
clc;

center_1 = [3, 4];
% center_2 = [5.5, 6.5];
center_2 = [7, 8];
num_point = [130, 150];
sparse_degree = [0.7, 1.0];

data_1 = get_gauss_data(center_1(1), center_1(2), sparse_degree(1), num_point(1));
label_1 = zeros(size(data_1, 1), 1);
data_2 = get_gauss_data(center_2(1), center_2(2), sparse_degree(2), num_point(2));
label_2 = ones(size(data_2, 1), 1);

subplot(121);
hold on;
scatter(data_1(:, 1), data_1(:, 2), 'r^');
scatter(data_2(:, 1), data_2(:, 2), 'bx');
title('ԭʼ����');
legend('������', '������')
hold off;

data = cat(1, data_1, data_2);
label = cat(1, label_1, label_2);

random_index = randperm(size(data, 1))';

train_data = data(random_index(1:floor(0.7 * size(data, 1))), :);
train_label = label(random_index(1:floor(0.7 * size(data, 1)), :));

test_data = data(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1)), :);
test_label = label(random_index(floor(0.7 * size(data, 1)) + 1:size(data, 1), :));

subplot(122);
hold on;
scatter(train_data(:, 1), train_data(:, 2), 'g');
scatter(test_data(:, 1), test_data(:, 2), 'y');
title('ѵ������');
legend('ѵ������', '��������');
hold off;

start = cputime;
model = svmtrain(train_label, train_data, '-c 1 -g 0.07');  %ѵ��ģ��
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_data, model);  %��ģ��Ԥ��
fprintf('training time:%.3f\n', cputime - start);