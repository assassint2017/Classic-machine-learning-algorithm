clear all;
clc;

%% 获取一批线性数据
data = get_line_data(6, 3, 3);
hold on;
data_x = data(:, 1);
data_y = data(:, 2);
scatter(data_x, data_y, 'r^');

%% 训练epsilon-SVR模型
model = svmtrain(data_y, data_x, '-s 3 -t 0 -c 1');  %训练模型，使用线性核函数
[predict_label, accuracy, decision_values] = svmpredict(data_y, data_x, model);  %用模型预测

%% 拟合结果展示
scatter(data_x, decision_values, 'b^');
title('SVM线性回归展示');
legend('原始数据', '拟合结果')
