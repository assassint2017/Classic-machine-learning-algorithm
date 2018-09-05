clear all;
clc

%% ���Ȼ�ȡһ����ά��ģ������
% ����һ������ѵ���ɣ�����һ�������ѣ�һ��������
point_center_x = [1, 3.5, 1.5, 5.5, 6];
point_center_y = [1, 2.5, 2.4, 7.5, 6];
num_points = [100, 90, 50, 110, 40];
sparse_degree = [0.5, 0.5, 0.5, 0.8, 0.75];

for i = 1:length(point_center_x)
    if i >= 4
        class_index = 1;
    else
        class_index = 0;
    end
    [tempdata, templabel] = get_data(point_center_x(i), point_center_y(i), num_points(i), sparse_degree(i), class_index);
    if i == 1
        data = tempdata;
        label = templabel;
    else
        data = cat(1, data, tempdata);
        label = cat(1, label, templabel);
    end
end

%% ��ԭʼ���ݽ��п��ӻ� 
set(gcf,'position',[300, 300, 1550, 450]);
figure(1);
colormap jet;
subplot(131);
scatter(data(:, 1), data(:, 2), 20, label);
title('ԭʼ����չʾ');
subplot(132);
for i = 1:length(num_points)
    if i == 1
        sublabel = ones(num_points(i), 1) * i - 1;
    else
        sublabel = cat(1, sublabel, ones(num_points(i), 1) * i - 1);
    end
end
hold on;
scatter(data(:, 1), data(:, 2), 20, sublabel);
scatter(point_center_x, point_center_y, 15, 'rx', 'linewidth', 15);
hold off
colorbar();
title('��������չʾ');

%% ����lvq����
subplot(133);

max_iter = 200000;  % �����㷨���ĵ�������
learning_rate = 1e-4;  % �����㷨��ѧϰ��
tol = 1e-3;  % �����㷨����������
num_tol = 20;  % �����㷨����ֹͣ����
sub_class = [3,2];  % �����㷨Ҫϸ�ֵ��������

[clustering_label, centroid] = lvq(data, label, sub_class, learning_rate, max_iter, tol, num_tol);

hold on;
scatter(data(:, 1), data(:, 2), 20, clustering_label);
scatter(centroid(:, 1), centroid(:, 2), 15, 'rx', 'linewidth', 15)
hold off;

colorbar();
title('LVQ����չʾ');

%% ��������ָ��
% ���ݽ��׼��Ԥ������ļ�������֮���MSE
point_center = cat(2, point_center_x', point_center_y');
MSE = sum((point_center - centroid) .^ 2, 2);
MSE = mean(MSE);
fprintf('������%f\n', MSE);