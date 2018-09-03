function centroid = K_means_getcentroid(data, pred_label, num_cluster)
%在当前的标签下获得新的质心坐标
for i =1:num_cluster
    temp = data(pred_label == i, :);
    if i == 1
        centroid = sum(temp, 1) ./size(temp, 1);
    else
        centroid = cat(1, centroid, sum(temp, 1) ./size(temp, 1));
    end
end

end