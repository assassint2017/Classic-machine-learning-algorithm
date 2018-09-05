function clustering_label = find_clustering_label(data, centroid)
% 根据质心坐标，对数据进行聚类，并返回聚类标签
num_class = size(centroid, 1);
for i = 1:num_class
    if i == 1
        distance = sum((data - centroid(i, :)) .^ 2, 2);
    else
        distance = cat(2, distance, sum((data - centroid(i, :)) .^ 2, 2));
    end
end
[~, clustering_label] = min(distance, [], 2);
clustering_label = clustering_label - 1;
end