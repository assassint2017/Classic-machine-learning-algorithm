function data = get_random_data(x,y,sparse_degree,num_point)
%xy对应一个点群的中央坐标点
%num_point代表点群的数量
%sparse_degree代表着一个点群的稀疏程度

x_point = x * ones(num_point, 1);
y_point = y * ones(num_point, 1);

data = cat(2, x_point, y_point) + randn(num_point, 2) * sparse_degree;
end