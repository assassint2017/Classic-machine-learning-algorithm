function pred_label = K_means_assignment(data, centroid, num_cluster)
%在当前的质心坐标下，对每一个数据点根据最小距离贴上标签

for i = 1:num_cluster
    temp_centroid = centroid(i, :);
    if i == 1
        distance = sqrt(sum((data - temp_centroid) .^2, 2));
    else
        distance = cat(2, distance, sqrt(sum((data - temp_centroid) .^2, 2)));
    end
end

[~, pred_label] = min(distance, [], 2);
end