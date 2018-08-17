function data = get_round_data(center_x, center_y, radius, num_point)
% ��һ���������ĵ��Բ�ڰ��վ��ȷֲ������ȡ����
random_radius = rand(num_point, 1) * radius;
random_angle = rand(num_point, 1) * 2 * pi;

x = random_radius .* cos(random_angle);
y = random_radius .* sin(random_angle);

x = x + center_x;
y = y + center_y;

data = cat(2, x, y);

end