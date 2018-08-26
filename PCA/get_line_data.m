function data = get_line_data(a, b, sparse_degree)
% 得到大体分布在一条直线周围的点群数据
% y=aX+b
x = -5:0.05:5;
y = a * x + b;

x = x';
y = y';

y = randn(length(y), 1) * sparse_degree + y;

data = cat(2, x ,y);

end