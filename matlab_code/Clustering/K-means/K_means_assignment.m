function pred_label = K_means_assignment(data, centroid, num_cluster)
%�ڵ�ǰ�����������£���ÿһ�����ݵ������С�������ϱ�ǩ

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