% libsvm��װ�̳� https://zhuanlan.zhihu.com/p/30485050
% ������һ���򵥵�SVM���Դ���
% �±ߵ�������libsvm�Ĺ���
% https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% ��ȥ�Ķ�һ���Դ���readme���й����ϵ���Ϣ�������Ķ����������ˣ��ǳ�ϸ�ڵĶ��������ṩ��˵��PDF

% libsvm�еĺ�������
% svmtrainѵ��������ѵ�����ݲ���ģ�͵�
% svmpredictѵ��������ʹ��ѵ����ģ��ȥԤ��������������
% libsvmread��Ҫ���ڶ�ȡ���� 
% libsvmwriteд���������ǰ���֪���ݴ����� 
% ���ṩ������������������һ���Ĵ��룬������Щ�Լ�Ҳ����Ū�����Բ��Ǳ���Ҫ����ѧϰ��
% �������Ҫ�Ļ���ǰ��������

clear all;
clc;
load heart_scale.mat  %���ز������ݼ�

% svmtrainǰһ��Ϊ��ǩ����һ��Ϊ���ݣ����Ϊ�����ǩ�����ݶ�����Ҫ��doubel���͵�
% ����ÿһ����һ�����ݣ��д����������ǩҲ��ͬ����ǩ��ȡֵ�Ƚ����⣬���Ǵ�0��ʼһ������һ������
model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');  %ѵ��ģ��
% svmpredict��һ���Ǳ�ǩ���ڶ��������ݣ���������ģ�ͣ����һ�����������
[predict_label, accuracy, decision_values] = svmpredict(heart_scale_label, heart_scale_inst, model);  %��ģ��Ԥ��

% svm��ѵ�����֮����ӡ���µ�һЩ��Ϣ
% optimization finished, #iter = 134 ��������
% nu = 0.433785 �����nu��nu-svm��һ����˼
% obj = -101.855060, rho = 0.426412 rho��bias���෴��
% nSV = 130, nBSV = 107
% Total nSV = 130 
% nSV:��׼֧����������,�����ڷ���ı߽��ϣ��ɳڱ�������0���ʸ�����ϵ�� 0=<ai<C  
% nBSV:�߽��֧����������,���ڷ���ı߽��ϣ��ɳڱ�������0����������ϵ�� ai = C

% �����������õ���һ��һ����ʽ
% The 'svmtrain' function returns a model which can be used for future
% prediction.  It is a structure and is organized as [Parameters, nr_class,
% totalSV, rho, Label, ProbA, ProbB, nSV, sv_coef, SVs]:
% 
%         -Parameters: parameters ������SVMģ�͵����ͣ���Ӧ-s -t -d -g -t�������
%         -nr_class: number of classes; = 2 for regression/one-class svm
%         -totalSV: total #SV  ֧�������ĸ���
%         -rho: -b of the decision function(s) wx+b ��Ӧbias
%         -Label: label of each class; empty for regression/one-class SVM
%         -sv_indices: values in [1,...,num_traning_data] to indicate SVs in the training set
%         -ProbA: pairwise probability information; empty if -b 0 or in one-class SVM
%         -ProbB: pairwise probability information; empty if -b 0 or in one-class SVM
%         -nSV: number of SVs for each class; empty for regression/one-class SVM
%         -sv_coef: coefficients for SVs in decision functions Ӧ�þ���ģ���еĲ���
%         -SVs: support vectors ֧�����������꣨�������ݵ㣩
% ����ֵaccuracy�ĵ�һ������ָʾ����Ч��������������ָʾ�ع�Ч��

% svmpredict�Ĳ�����Ծ����˺ֻܶ࣬��-b��-q
% libsvm_options:
%     -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet
%     ʹ��-b��ǰ����ģ����ѵ���Ĺ�����-bѡ����Ѿ���ѡ��
%     -q : quiet mode (no outputs)

% ��̳��һ����������ӣ���ǿ���������SVM�������������⣬���Բμ�����˵���������
% faruto����
% http://www.matlabsky.com/thread-12649-1-1.html
