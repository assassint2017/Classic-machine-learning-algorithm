function [clustering_label, centroid] = lvq(data, label, sub_class, learning_rate, max_iter, tol, num_tol)
% Learning Vector Quantization ѧϰ������������
% �㷨Ŀǰ���ڶ�ά���������ģ�������չ����άҲ�Ƿǳ��򵥵�����
% data ��������
% label ���ݵ�ԭʼ�����ǩ
% sub_class Ҫϸ�ֵ�������
% learning_rate ѧϰ��
% max_iter �㷨���ĵ�������
% tol �㷨��������������ÿһ������������ά���ϵ���Сλ��
% num_tol ֻ��num_tol�ֶ��������ֹͣ�������㷨�Ż���ǰ��ֹ

% clustering_label ����ϸ�ֵ����ݱ�ǩ

%% ���Ƚ��д��������ʼ��
% ��ÿһ��Ԥ����֪������������ѡ���������������Ϊ��ʼ��
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

% ���������ʼ�������ģ������ݽ��о��࣬���õ���ʼ�ľ����ǩ
% clustering_label = find_clustering_label(data, centroid);

%% ��ʼ���е���
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
        fprintf('����������%d\n', i);
        disp('�㷨����!');
        break;
    end
   
    update_vector = update_vector * learning_rate;
   
    centroid(min_distance_index, :) = centroid(min_distance_index, :) + update_vector;
    
    % һ�ֵ�����������ʼ������һ�ֵ���
end

%% ��������
% �����������꣬���ؾ����ǩ
clustering_label = find_clustering_label(data, centroid);

end