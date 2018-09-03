"""

使用KNN算法对iris数据集进行分类
"""

import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn.externals import joblib
import sklearn.model_selection as ms
from sklearn.neighbors import KNeighborsClassifier


iris = datasets.load_iris()
data = iris.data  # 获取一个数据集的数据
label = iris.target  # 获取一个数据集的标签

"""
iris 鳶尾花 数据集有150个数据 用于多分类学习
每个数据有四个特征,代表着一个花朵的属性,最终的目标是分对花朵的类别
一共有三种类别的花
"""

train_data, test_data, train_label, test_lable = ms.train_test_split(data, label, test_size=0.3)

knn = KNeighborsClassifier(7)  # 在sk中创建model就是实例化一个对象,当然在初始化的过程中,
                               # 可以赋予不同的参数,但是通常情况下,sk自带的默认参数就很不错

# 进行交叉验证选择最近邻的数量
max_neighbor = 31
neighbor_list = list(range(1, max_neighbor + 1))
score_list = []

for num_neighbor in neighbor_list:
    knn = KNeighborsClassifier(num_neighbor)
    score = ms.cross_val_score(knn, train_data, train_label, scoring='accuracy', cv=5)
    score_list.append(score.mean())

plt.plot(neighbor_list, score_list)
plt.xlabel('number of neighbor')
plt.ylabel('acc')
plt.title('cv for number of neighbor')

max_score = max(score_list)
max_socre_index = score_list.index(max_score)
best_neighbor = neighbor_list[max_socre_index]
print('best number of neighbor', neighbor_list[max_socre_index])
print(score_list)
print(neighbor_list)

# 选择好参数之后,在整个训练集上进行训练
knn = KNeighborsClassifier(best_neighbor)
knn.fit(train_data, train_label)  # fit函数用于拟合数据

# 最后再测试集上得到测试精度
pred_label = knn.predict(test_data)  # predict函数用于预测数据
print('acc:{:.3f}%'.format((pred_label == test_lable).sum() / test_lable.shape[0] * 100))
plt.show()

# 保存模型
# 保存模型可以使用sk自带的joblib或者python自带的pickle,但是使用joblib可能会快一点
joblib.dump(knn, './knn.pkl')

# 恢复模型
knn = joblib.load('./knn.pkl')

# 使用恢复好的模型,在测试集上进行测试
pred_label = knn.predict(test_data)  # predict函数用于预测数据
print('restore model acc:{:.3f}%'.format((pred_label == test_lable).sum() / test_lable.shape[0] * 100))
