clear all;
clc;

%% ��ȡһ������������
hold on;
data_x = 1:0.1:10
data_y = data_x .^3 - 8 * data_x .^2;

data_x = data_x';
data_y = data_y';
scatter(data_x, data_y, 'r^');

%% ѵ��epsilon-SVRģ��
model = svmtrain(data_y, data_x, '-s 3 -t 2 -c 100 -g 0.5');  %ѵ��ģ�ͣ�ʹ�ø�˹�˺���
[predict_label, accuracy, decision_values] = svmpredict(data_y, data_x, model);  %��ģ��Ԥ��

%% ��Ͻ��չʾ
scatter(data_x, decision_values, 'bo');
title('SVM�����Իع�չʾ');
legend('ԭʼ����', '��Ͻ��')
