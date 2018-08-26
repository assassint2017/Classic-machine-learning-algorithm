clear all;
clc;

% ����PCA�㷨������ά����ѹ����һά����
data = get_line_data(6, 3, 3);

hold on;
scatter(data(:, 1), data(:, 2), 'r^');
title('PCA');

% ����������ֵ��һ��
mean = sum(data, 1) ./ length(data);
data = data - mean;

% ���ȼ���Э�������
gamma = (data' * data) / length(data);

[U, S, V] = svd(gamma);

U = U(:, 1);
compressd_data = U' * data';
compressd_data = compressd_data';

reconstruction_data = U * compressd_data';
reconstruction_data = reconstruction_data';
reconstruction_data = reconstruction_data + mean;

scatter(reconstruction_data(:, 1), reconstruction_data(:, 2), 'bx');
legend('ԭʼ����', '�ؽ�����');
hold off;
