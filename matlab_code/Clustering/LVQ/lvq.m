function [clustering_label, centroid] = lvq(data, label, sub_class, learning_rate, max_iter, tol, num_tol)
% Learning Vector Quantization 学习向量量化聚类
% 算法目前是在二维数据下做的，但是扩展到多维也是非常简单的事情
% data 输入数据
% label 数据的原始大类标签
% sub_class 要细分的类别个数
% learning_rate 学习率
% max_iter 算法最大的迭代次数
% tol 算法的收敛条件，既每一个簇心在所有维度上的最小位移
% num_tol 只有num_tol轮都满足迭代停止条件，算法才会提前终止

% clustering_label 返回细分的数据标签

%% 首先进行簇心随机初始化
% 在每一个预先已知的类别中随机挑选子类个数的质心作为初始点
num_class = max(label) + 1;
num_sub_class = sum(sub_class);

for i = 1:num_class
    temp_data = data(label == (i - 1), :);
    random_index = randperm(size(temp_data, 1));
    if i == 1
        centroid = temp_data(random_index(1:sub_class(i)), :);
        centroid_label = ones(sub_class(i), 1) * i - 1;
    else
        centroid = cat(1, centroid, temp_data(random_index(1:sub_class(i)), :));
        centroid_label = cat(1, centroid_label, ones(sub_class(i), 1) * i - 1);
    end
end

% 根据随机初始化的质心，对数据进行聚类，并得到初始的聚类标签
% clustering_label = find_clustering_label(data, centroid);

%% 开始进行迭代
num_convergence = 0;

for i = 1:max_iter
    random_index = randperm(size(data, 1));
    temp_data = data(random_index(1), :);
    temp_label = label(random_index(1));
   
    distance = sum((centroid - temp_data) .^ 2, 2);
    [~, min_distance_index] = min(distance, [], 1);
   
    update_vector = temp_data - centroid(min_distance_index, :);
   
    if centroid_label(min_distance_index) ~= temp_label
        update_vector = update_vector * -1; 
    end
   
    if update_vector <= tol
        num_convergence = num_convergence + 1;
    else
        num_convergence = 0;
    end
    
    if num_convergence == num_tol
        fprintf('迭代轮数：%d\n', i);
        disp('算法收敛!');
        break;
    end
   
    update_vector = update_vector * learning_rate;
   
    centroid(min_distance_index, :) = centroid(min_distance_index, :) + update_vector;
    
    % 一轮迭代结束，开始进入下一轮迭代
end

%% 迭代结束
% 根据质心坐标，返回聚类标签
clustering_label = find_clustering_label(data, centroid);

end