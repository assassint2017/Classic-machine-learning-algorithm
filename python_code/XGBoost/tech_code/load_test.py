import xgboost as xgb

import sklearn.metrics as metrics

# 读取测试数据
test_matrix = xgb.DMatrix('./test.buffer')

# 查看数据情况
print(test_matrix.num_row())  # 查看数据量
print(test_matrix.num_col())  # 查看数据特征维度

# 恢复模型
clf = xgb.Booster()
clf.load_model('./test.model')

# 测试精度
pred_label = clf.predict(test_matrix)
acc = metrics.accuracy_score(test_matrix.get_label(), pred_label)

# 如果是二分类,还可以计算如下的一些评价指标
# precision = metrics.precision_score(test_matrix.get_label(), pred_label)
# recall = metrics.recall_score(test_matrix.get_label(), pred_label)
# f1 = metrics.f1_score(test_matrix.get_label(), pred_label)

print('acc:', acc)

# print('precision:', precision)
# print('recall:', recall)
# print('f1:', f1)
