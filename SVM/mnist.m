clear all;
clc;

start = cputime;
[train_data, train_label, test_data, test_label] = get_mnist_data();
fprintf('time of get data:%.3f s\n', cputime - start);

start = cputime;
model = svmtrain(train_label, train_data, '-c 1 -g 0.07');  %ѵ��ģ��
[predict_label, accuracy, dec_values] = svmpredict(test_label, test_data, model);  %��ģ��Ԥ��
fprintf('time of training:%.3f min\n', (cputime - start) / 60);