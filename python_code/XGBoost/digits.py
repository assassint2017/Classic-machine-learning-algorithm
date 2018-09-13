"""

使用sklearn API
"""

import xgboost as xgb
import numpy as np
import matplotlib.pyplot as plt

from sklearn import datasets
import sklearn.model_selection as ms

# 从sk中获取数据,对于分类问题,需要标签从0开始递增
digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=1)

# 转换成xgb支持的数据类型
train_matrix = xgb.DMatrix(train_data, train_label)
test_matrix = xgb.DMatrix(test_data, test_label)

# 定义参数 0.9665871121718377
num_trees = 90
learning_rate = 0.5
max_depth = 3
colsample_bytree = 0.2

params_grid = {'reg_lambda': np.linspace(0.5, 5.0, 10)}

# 进行网格搜索寻找参数
clf = ms.GridSearchCV(xgb.XGBClassifier(n_estimators=num_trees,
                                        max_depth=max_depth,
                                        learning_rate=learning_rate,
                                        colsample_bytree=colsample_bytree,
                                        random_state=1),
                      params_grid, n_jobs=-1, cv=10, scoring='accuracy')

clf.fit(train_data, train_label)

# 打印最佳参数
print(clf.best_params_)
print(clf.best_score_)

plt.plot(clf.cv_results_['param_reg_lambda'],
         clf.cv_results_['mean_test_score'],
         '-o')

plt.show()
