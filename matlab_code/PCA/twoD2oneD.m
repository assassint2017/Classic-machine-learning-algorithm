clear all;
clc;

% 利用PCA算法，将二维数据压缩成一维数据
data = get_line_data(6, 3, 3);

hold on;
scatter(data(:, 1), data(:, 2), 'r^');
title('PCA');

% 进行特征均值归一化
mean = sum(data, 1) ./ length(data);
data = data - mean;

% 首先计算协方差矩阵
gamma = (data' * data) / length(data);

[U, S, V] = svd(gamma);

U = U(:, 1);
compressd_data = U' * data';
compressd_data = compressd_data';

reconstruction_data = U * compressd_data';
reconstruction_data = reconstruction_data';
reconstruction_data = reconstruction_data + mean;

scatter(reconstruction_data(:, 1), reconstruction_data(:, 2), 'bx');
legend('原始数据', '重建数据');
hold off;
