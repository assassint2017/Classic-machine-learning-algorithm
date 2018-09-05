function [data, label] = get_data(x, y, num_point, sparse_degree, class_index)
% ��ȡ���ݵĺ����ļ�
% x,y ���ݵ����ĵ�����
% num_point ���ݸ���
% sparse_degree ������ɢ�̶�
% class_index ������
X = ones(num_point, 1) * x + sparse_degree * randn(num_point, 1);
Y = ones(num_point, 1) * y + sparse_degree * randn(num_point, 1);

data = cat(2, X, Y);
label = class_index * ones(num_point, 1);
end