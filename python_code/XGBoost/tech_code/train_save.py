import xgboost as xgb
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

test_matrix.save_binary('./test.buffer')  # 将文件保存为xgboost的二进制buffer文件,这样读取的时候速度更快

# 定义参数
num_trees = 300
params = {'eta': 0.5,
          'max_depth': 3,
          'objective': 'multi:softmax',
          'num_class': 10,
          'lambda': 1,
          'alpha': 0,
          'subsample': 0.5,
          'colsample_bytree': 0.8,
          'tree_method': 'gpu_hist',
          'eval_metric': 'merror',
          'seed': 1,
          'silent': 1}

# 训练模型
clf = xgb.train(params, train_matrix, num_boost_round=num_trees,
                evals=[(train_matrix, 'train'), (test_matrix, 'eval')],
                early_stopping_rounds=20)

print(clf.best_iteration)
# eval = xgb.cv(params, train_matrix, num_trees, 10)
# print(eval)

# 直接测试精度
pred_label = clf.predict(test_matrix)

# 可视化模型
# xgb.plot_tree(clf, num_trees=5)
xgb.plot_importance(clf, importance_type='gain', max_num_features=20)
plt.show()

# 保存模型
clf.save_model('./test.model')

