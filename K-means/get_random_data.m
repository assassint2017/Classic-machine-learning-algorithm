function data = get_random_data(x,y,sparse_degree,num_point)
%xy��Ӧһ����Ⱥ�����������
%num_point�����Ⱥ������
%sparse_degree������һ����Ⱥ��ϡ��̶�

x_point = x * ones(num_point, 1);
y_point = y * ones(num_point, 1);

data = cat(2, x_point, y_point) + randn(num_point, 2) * sparse_degree;
end