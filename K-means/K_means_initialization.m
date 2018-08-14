function centroid = K_means_initialization(data, num_cluster)
%Ѱ�ҳ�ʼ�����ĵ����꣬������������Ļ���ѧϰ�γ�
%��ʼ���ķ�ʽ�Ǵ����ݵ��������ѡ����
centroid = zeros(num_cluster, 2);
row_index = randsrc(num_cluster, 1, 1:size(data, 1));
for i = 1:num_cluster
    centroid(i, :) = data(row_index(i), :);
end
end