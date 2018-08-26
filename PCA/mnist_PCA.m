% ��MNIST����ͶӰ����άƽ����
clear all;
clc;

%% ��ȡMNIST����
% [data, label, ~, ~] = get_mnist_data();
[~, ~, data, label] = get_mnist_data();

%% ������һ��
mean = sum(data, 1) ./ length(data);
std = sqrt(sum((data - mean ).^2, 1) ./ length(data));

data = (data - mean) ./ (std + 1e-5);

%% PCA��ά
% ����Э�������
gamma = (data' * data) ./ length(data);

% ��������ֵ�ֽ�
[U, S, V] = svd(gamma);

% U = U(:, 1:2);
U = U(:, 1:3);

compressed_data = U' * data';
compressed_data = compressed_data';

% scatter(compressed_data(:, 1), compressed_data(:, 2), 30, label);
scatter3(compressed_data(:, 1), compressed_data(:, 2), compressed_data(:, 3), 30, label);
colormap jet;
colorbar();
