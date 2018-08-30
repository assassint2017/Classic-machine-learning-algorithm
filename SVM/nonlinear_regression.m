clear all;
clc;

%% 获取一批非线性数据
hold on;
data_x = 1:0.1:10
data_y = data_x .^3 - 8 * data_x .^2;

data_x = data_x';
data_y = data_y';
scatter(data_x, data_y, 'r^');

%% 训练epsilon-SVR模型
model = svmtrain(data_y, data_x, '-s 3 -t 2 -c 100 -g 0.5');  %训练模型，使用高斯核函数
[predict_label, accuracy, decision_values] = svmpredict(data_y, data_x, model);  %用模型预测

%% 拟合结果展示
scatter(data_x, decision_values, 'bo');
title('SVM非线性回归展示');
legend('原始数据', '拟合结果')
