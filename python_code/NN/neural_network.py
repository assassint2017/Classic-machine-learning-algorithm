from sklearn import neural_network as nn
from sklearn import datasets
import sklearn.model_selection as ms

# 获取数据集
digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=3)

# 利用网格搜索来寻找最优参数
parameter_grid = {'hidden_layer_sizes': [1, 2, 3],
                  'activation': ['logistic', 'tanh', 'relu'],
                  'learning_rate': ['constant', 'invscaling', 'adaptive']}

clf = ms.GridSearchCV(nn.MLPClassifier(max_iter=1500), parameter_grid, 'accuracy', cv=10, n_jobs=-1)
# n_jobs并行运行的任务数，-1表示使用所有CPU

clf.fit(train_data, train_label)

for key, value in clf.cv_results_.items():
    print(key, value)
    print('----------')

print('best_params:', clf.best_params_)  # 打印最好的参数组合
print('best acc:', clf.best_score_)  # 根据之前设定好的评价标准,打印最高值
print('metric', clf.scorer_)  # 打印之前设定好的评价标准

# 由于GridSearchCV中refit默认为True,也就是获得最优参数之后会在整个训练集中训练,因此使用起来十分方便
net = clf.best_estimator_  # 获得最优参数组合下的模型
pred_label = net.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0] * 100))

# test acc:80.000%
