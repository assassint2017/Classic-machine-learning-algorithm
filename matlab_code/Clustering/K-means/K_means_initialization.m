function centroid = K_means_initialization(data, num_cluster)
%寻找初始的蔟心点坐标，这里根据吴恩达的机器学习课程
%初始化的方式是从数据点中随机挑选即可
centroid = zeros(num_cluster, 2);
row_index = randsrc(num_cluster, 1, 1:size(data, 1));
for i = 1:num_cluster
    centroid(i, :) = data(row_index(i), :);
end
end