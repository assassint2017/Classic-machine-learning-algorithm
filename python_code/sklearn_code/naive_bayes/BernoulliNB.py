"""

朴素贝叶斯分类器中的伯努利模式
用于处理二值特征
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn import naive_bayes
from sklearn import datasets
import sklearn.model_selection as ms

digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=1)

binarize_range = np.linspace(0, 16, 17)

# # 利用交叉验证来选择最佳的二值化阈值
# train_scores, val_scores = ms.validation_curve(naive_bayes.BernoulliNB(), train_data, train_label, 'binarize',
#                                                binarize_range, cv=10,
#                                                scoring='accuracy')
#
# train_scores = train_scores.mean(axis=1)
# test_scores = val_scores.mean(axis=1)
#
# plt.plot(binarize_range, train_scores, 'o-', label='train acc')
# plt.plot(binarize_range, test_scores, 'o-', label='test acc')
# plt.xlabel('binarize threshold')
# plt.ylabel('acc')
# plt.legend(loc='best')
# plt.show()

# 选择好参数之后就可以在整个训练集上进行训练,并在测试集上进行测试
bayes = naive_bayes.BernoulliNB(binarize=7)
bayes.fit(train_data, train_label)
pred_label = bayes.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0] * 100))

# test acc:89.259%
