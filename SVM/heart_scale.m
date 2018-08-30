% libsvm安装教程 https://zhuanlan.zhihu.com/p/30485050
% 下面是一个简单的SVM测试代码
% 下边的连接是libsvm的官网
% https://www.csie.ntu.edu.tw/~cjlin/libsvm/
% 多去阅读一下自带的readme还有官网上的信息，基本的东西都包含了，非常细节的东西官网提供了说明PDF

% libsvm中的函数介绍
% svmtrain训练函数，训练数据产生模型的
% svmpredict训练函数，使用训练的模型去预测来的数据类型
% libsvmread主要用于读取数据 
% libsvmwrite写函数，就是把已知数据存起来 
% 还提供了网格搜索和特征归一化的代码，但是这些自己也可以弄，所以不是必须要进行学习的
% 因此最重要的还是前两个函数

clear all;
clc;
load heart_scale.mat  %加载测试数据集

% svmtrain前一个为标签，后一个为数据，最后为命令，标签和数据都必须要是doubel类型的
% 数据每一行是一个数据，列代表个数，标签也是同理，标签的取值比较随意，但是从0开始一个类别加一就行了
model = svmtrain(heart_scale_label, heart_scale_inst, '-c 1 -g 0.07');  %训练模型
% svmpredict第一个是标签，第二个是数据，第三个是模型，最后一个是命令参数
[predict_label, accuracy, decision_values] = svmpredict(heart_scale_label, heart_scale_inst, model);  %用模型预测

% svm在训练结果之后会打印以下的一些信息
% optimization finished, #iter = 134 迭代次数
% nu = 0.433785 这里的nu和nu-svm是一个意思
% obj = -101.855060, rho = 0.426412 rho是bias的相反数
% nSV = 130, nBSV = 107
% Total nSV = 130 
% nSV:标准支持向量个数,就是在分类的边界上，松弛变量等于0，朗格朗日系数 0=<ai<C  
% nBSV:边界的支持向量个数,不在分类的边界上，松弛变量大于0，拉格郎日系数 ai = C

% 多分类问题采用的是一对一的形式
% The 'svmtrain' function returns a model which can be used for future
% prediction.  It is a structure and is organized as [Parameters, nr_class,
% totalSV, rho, Label, ProbA, ProbB, nSV, sv_coef, SVs]:
% 
%         -Parameters: parameters 代表着SVM模型的类型，对应-s -t -d -g -t五个参数
%         -nr_class: number of classes; = 2 for regression/one-class svm
%         -totalSV: total #SV  支持向量的个数
%         -rho: -b of the decision function(s) wx+b 对应bias
%         -Label: label of each class; empty for regression/one-class SVM
%         -sv_indices: values in [1,...,num_traning_data] to indicate SVs in the training set
%         -ProbA: pairwise probability information; empty if -b 0 or in one-class SVM
%         -ProbB: pairwise probability information; empty if -b 0 or in one-class SVM
%         -nSV: number of SVs for each class; empty for regression/one-class SVM
%         -sv_coef: coefficients for SVs in decision functions 应该就是模型中的参数
%         -SVs: support vectors 支持向量的坐标（部分数据点）
% 返回值accuracy的第一个用于指示分类效果，后两个用于指示回归效果

% svmpredict的参数相对就少了很多，只有-b和-q
% libsvm_options:
%     -b probability_estimates: whether to predict probability estimates, 0 or 1 (default 0); one-class SVM not supported yet
%     使用-b的前提是模型在训练的过程中-b选项就已经勾选了
%     -q : quiet mode (no outputs)

% 论坛上一个大神大帖子，贼强，如果对于SVM还可以其他问题，可以参见这个人的其他帖子
% faruto大神
% http://www.matlabsky.com/thread-12649-1-1.html
