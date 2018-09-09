"""

使用梯度提升树来完成回归
"""

import sklearn.model_selection as ms
from sklearn import datasets
from sklearn import ensemble

# 获取数据集
boston = datasets.load_boston()
data = boston.data
label = boston.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=3)

# 构建模型
clf = ensemble.GradientBoostingRegressor()
clf.fit(train_data, train_label)

print(clf.feature_importances_)
print(sum(clf.feature_importances_))

pred_label = clf.predict(test_data)
print('MSE:{:.3f}'.format(((pred_label - test_label) ** 2).mean()))
