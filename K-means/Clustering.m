clc;
clear all;

%% �����㷨������
num_iteration = 200; % �㷨��������
num_cluster = 5; % �������� 
min_sparse_degree = 0.5; % ������Сϡ��̶�
max_spares_degree = 5; % �������ϡ��̶�
min_num_point = 50; % ÿ����Ⱥ��С�ĵ�����
max_num_point = 150; % ÿ����Ⱥ���ĵ�����
min_number = 5; % ��Сֵ
max_number = 35; % ���ֵ

num_points = zeros(num_cluster, 1);
for i = 1:num_cluster
    num_points(i) = randi([min_num_point, max_num_point]);
end

sparse_degree = zeros(num_cluster, 1);
for i = 1:num_cluster
    sparse_degree(i) = min_sparse_degree + (max_spares_degree - min_sparse_degree) * rand(1, 1);
end

%% ��ȡ����
for i = 1:num_cluster
    temp = cat(2, randi([min_number, max_number], 1, 1) * ones(num_points(i), 1), randi([min_number, max_number], 1, 1) * ones(num_points(i), 1)); 
    temp = temp + randn(num_points(i), 2) * sparse_degree(i);
    if i == 1
        data = temp;
    else
        data = cat(1, data, temp);
    end
end

for i = 1:num_cluster
    temp = i * ones(num_points(i), 1);
    if i == 1
        label = temp;
    else
        label = cat(1, label, temp);
    end
end

%% չʾԭʼ����
set(gcf,'position',[300, 300, 1450, 450])
figure(1);
grid on;
subplot(131);
scatter(data(:,1), data(:,2), 30);
title('ԭʼ����');
xlabel('x');
ylabel('y');
subplot(132);
scatter(data(:,1), data(:,2), 30, label);
colorbar();
title('���׼');
xlabel('x');
ylabel('y');

%% K-means����
%% �������������ʼ��
subplot(133)
hold on

centroid = K_means_initialization(data, num_cluster);
pred_label = K_means_assignment(data, centroid, num_cluster);

scatter(data(:,1), data(:,2), 30, pred_label);
scatter(centroid(:,1), centroid(:,2), 200, 'rx', 'LineWidth',2.5);

colorbar();
title('���׼');
xlabel('x');
ylabel('y');

%% ��ʼ�����Ż�
for i = 1:num_iteration
    centroid = K_means_getcentroid(data, pred_label, num_cluster);
    pred_label = K_means_assignment(data, centroid, num_cluster);
    
    if i < 20
        pause(1);
    else
        pause(0.01);
    end
    
    cla;
    scatter(data(:,1), data(:,2), 30, pred_label);
    scatter(centroid(:,1), centroid(:,2), 200, 'rx', 'LineWidth',2.5);
    legend(sprintf('iteration:%d', i));
end