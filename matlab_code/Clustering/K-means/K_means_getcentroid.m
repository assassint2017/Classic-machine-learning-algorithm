function centroid = K_means_getcentroid(data, pred_label, num_cluster)
%�ڵ�ǰ�ı�ǩ�»���µ���������
for i =1:num_cluster
    temp = data(pred_label == i, :);
    if i == 1
        centroid = sum(temp, 1) ./size(temp, 1);
    else
        centroid = cat(1, centroid, sum(temp, 1) ./size(temp, 1));
    end
end

end