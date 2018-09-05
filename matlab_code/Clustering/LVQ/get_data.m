function [data, label] = get_data(x, y, num_point, sparse_degree, class_index)
% 获取数据的函数文件
% x,y 数据的中心点坐标
% num_point 数据个数
% sparse_degree 数据离散程度
% class_index 类别序号
X = ones(num_point, 1) * x + sparse_degree * randn(num_point, 1);
Y = ones(num_point, 1) * y + sparse_degree * randn(num_point, 1);

data = cat(2, X, Y);
label = class_index * ones(num_point, 1);
end