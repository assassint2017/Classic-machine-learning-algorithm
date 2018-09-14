import csv

import xgboost as xgb
import matplotlib.pyplot as plt
import sklearn.model_selection as ms

from utility import get_data


# 获取数据
data, label = get_data(training=True)
train_data, val_data, train_label, val_label = ms.train_test_split(data, label, test_size=0.25, random_state=1)

feature_name = ['Pclass', 'Sex', 'Fare', 'is_child', 'family_size', 'Embarked-c', 'Embarked-s', 'Embarked-q']

data_matrix = xgb.DMatrix(data, label, feature_names=feature_name)
train_matrix = xgb.DMatrix(train_data, train_label, feature_names=feature_name)
val_matrix = xgb.DMatrix(val_data, val_label, feature_names=feature_name)
test_matrix = xgb.DMatrix(get_data(training=False), feature_names=feature_name)


num_trees = 47
eval_list = [(train_matrix, 'train'), (val_matrix, 'eval')]
params = {
          'objective': 'reg:logistic',
          'eval_metric': 'error',
          'max_depth': 5,
          'min_child_weight': 1,
          'eta': 0.5,
          'subsample': 1,
          'colsample_bytree': 1,
          'colsample_bylevel': 1,
          'lambda': 1,
          'alpha': 0,
          'silent': 1,
          'seed': 1}

clf = xgb.cv(params, data_matrix, num_trees, 5,
             # early_stopping_rounds=20,
             seed=4,
             verbose_eval=True)


# 调整好参数后,在整个训练集上进行训练
clf = xgb.train(params, data_matrix, num_trees)
pred_label = clf.predict(test_matrix)

xgb.plot_importance(clf, importance_type='gain')
plt.show()

# 对测试集进行预测
# 将预测结果写入到csv文件中
with open('pred.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['PassengerId', 'Survived'])

    for index, label in enumerate(pred_label, start=892):
        if label >= 0.5:
            label = 1
        else:
            label = 0

        line = [str(index), str(label)]
        writer.writerow(line)
