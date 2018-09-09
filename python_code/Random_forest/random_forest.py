import sklearn.model_selection as ms
from sklearn import datasets
from sklearn import ensemble

# 获取数据集
digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=3)

clf = ensemble.RandomForestClassifier(20, max_features='log2', oob_score=True, random_state=4)
clf.fit(train_data, train_label)

print(clf.oob_score_)
print(clf.oob_decision_function_)
print(clf.feature_importances_)
print(sum(clf.feature_importances_))

pred_label = clf.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0] * 100))
