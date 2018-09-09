"""

sklearn.tree下边的sklearn.tree.ExtraTreeClassifier
是sklearn.ensemble.ExtraTreesClassifier的基类,所以使用的时候,还是用集成方法下边的

这个算法是随机森林的一种变体
"""

import sklearn.model_selection as ms
from sklearn import datasets
from sklearn import ensemble

# 获取数据集
digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=3)

clf = ensemble.ExtraTreesClassifier(20)
clf.fit(train_data, train_label)


print(clf.feature_importances_)
print(sum(clf.feature_importances_))

# 在sk中,如果取样方式不是boot...的话,就没法计算oob误差,并不像某一个帖子说的可以用全部训练样本作为oob样本

pred_label = clf.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0] * 100))
