import matplotlib.pyplot as plt

import sklearn.tree as tree
import sklearn.model_selection as ms
from sklearn import datasets

# 获取数据集
digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=3)

# 网格搜索进行参数选择
parameter_grid = {
    'splitter': ['best', 'random'],
    'max_depth': list(range(3, 9)),
    'min_samples_split': list(range(2, 11)),
    'min_samples_leaf': list(range(1, 11)),
    'max_features': [None, 'auto', 'sqrt']
}

clf = ms.GridSearchCV(tree.DecisionTreeClassifier(), parameter_grid, scoring='accuracy', n_jobs=-1, cv=5)
clf.fit(train_data, train_label)

print(clf.best_params_)
print(clf.best_score_)
print(clf.best_estimator_.feature_importances_)

# 展示每个特征的重要程度
plt.scatter(list(range(1, len(clf.best_estimator_.feature_importances_) + 1)),
            clf.best_estimator_.feature_importances_)

# 下边这个链接讲决策树的各种参数的,可以看到,当特征数量非常多的时候,splitter,max_features有效的加快了速度
# 并防止了过拟合
# https://www.cnblogs.com/pinard/p/6056319.html
# 目前的sklearn还没有实现后剪枝,不过预剪枝倒是有不少
# tree.export_graphviz(clf.best_estimator_, './tree.dot', rounded=True, filled=True, rotate=True, max_depth=3)

pred_label = clf.best_estimator_.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0] * 100))

plt.show()

