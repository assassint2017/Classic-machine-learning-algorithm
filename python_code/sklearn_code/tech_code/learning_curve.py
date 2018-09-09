import numpy as np
import matplotlib.pyplot as plt

import sklearn.model_selection as ms
from sklearn import datasets
from sklearn import svm


digits = datasets.load_digits()
data = digits.data
label = digits.target

print(data.shape)
print(label.shape)

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3)

clf = svm.SVC(gamma=1e-3)

# 学习曲线用来检查模型出现了过拟合还是欠拟合问题
train_sizes, train_scores, test_scores = ms.learning_curve(clf, train_data, train_label,
                                                           train_sizes=np.linspace(0.1, 1, 20),
                                                           cv=10, scoring='accuracy')

print(train_sizes)
print(train_sizes.shape)
print(train_scores.shape)
print(test_scores.shape)

train_scores = train_scores.mean(axis=1)
test_scores = test_scores.mean(axis=1)

plt.plot(train_sizes, train_scores, label='train acc')
plt.plot(train_sizes, test_scores, label='test acc')
plt.xlabel('train sizes')
plt.ylabel('acc')
plt.legend(loc='best')
plt.show()
