clc;
clear all;

%% 定义算法超参数
num_iteration = 200; % 算法迭代次数
num_cluster = 5; % 蔟的数量 
min_sparse_degree = 0.5; % 蔟的最小稀疏程度
max_spares_degree = 5; % 蔟的最大稀疏程度
min_num_point = 50; % 每个蔟群最小的点数量
max_num_point = 150; % 每个蔟群最大的点数量
min_number = 5; % 最小值
max_number = 35; % 最大值

num_points = zeros(num_cluster, 1);
for i = 1:num_cluster
    num_points(i) = randi([min_num_point, max_num_point]);
end

sparse_degree = zeros(num_cluster, 1);
for i = 1:num_cluster
    sparse_degree(i) = min_sparse_degree + (max_spares_degree - min_sparse_degree) * rand(1, 1);
end

%% 获取数据
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

%% 展示原始数据
set(gcf,'position',[300, 300, 1450, 450])
figure(1);
grid on;
subplot(131);
scatter(data(:,1), data(:,2), 30);
title('原始数据');
xlabel('x');
ylabel('y');
subplot(132);
scatter(data(:,1), data(:,2), 30, label);
colorbar();
title('金标准');
xlabel('x');
ylabel('y');

%% K-means聚类
%% 对蔟心做随机初始化
subplot(133)
hold on

centroid = K_means_initialization(data, num_cluster);
pred_label = K_means_assignment(data, centroid, num_cluster);

scatter(data(:,1), data(:,2), 30, pred_label);
scatter(centroid(:,1), centroid(:,2), 200, 'rx', 'LineWidth',2.5);

colorbar();
title('金标准');
xlabel('x');
ylabel('y');

%% 开始迭代优化
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