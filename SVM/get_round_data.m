function data = get_round_data(center_x, center_y, radius, num_point)
% 从一个给定中心点的圆内按照均匀分布随机提取数据
random_radius = rand(num_point, 1) * radius;
random_angle = rand(num_point, 1) * 2 * pi;

x = random_radius .* cos(random_angle);
y = random_radius .* sin(random_angle);

x = x + center_x;
y = y + center_y;

data = cat(2, x, y);

end