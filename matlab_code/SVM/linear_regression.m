clear all;
clc;

%% ��ȡһ����������
data = get_line_data(6, 3, 3);
hold on;
data_x = data(:, 1);
data_y = data(:, 2);
scatter(data_x, data_y, 'r^');

%% ѵ��epsilon-SVRģ��
model = svmtrain(data_y, data_x, '-s 3 -t 0 -c 1');  %ѵ��ģ�ͣ�ʹ�����Ժ˺���
[predict_label, accuracy, decision_values] = svmpredict(data_y, data_x, model);  %��ģ��Ԥ��

%% ��Ͻ��չʾ
scatter(data_x, decision_values, 'b^');
title('SVM���Իع�չʾ');
legend('ԭʼ����', '��Ͻ��')
