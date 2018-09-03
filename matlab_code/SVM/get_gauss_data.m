function data = get_gauss_data(x,y,sparse_degree,num_point)
%xy对应一个点群的中央坐标点
%num_point代表点群的数量
%sparse_degree代表着一个点群的稀疏程度
%返回的是一个符合高斯分布的点群

x_point = x * ones(num_point, 1);
y_point = y * ones(num_point, 1);

data = cat(2, x_point, y_point) + randn(num_point, 2) * sparse_degree;
end