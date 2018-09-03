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

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=3)

clf = svm.SVC()
gamma_range = np.logspace(-3.5, -2.5, 10)

# 验证曲线用来选择模型中的某一个参数
train_scores, test_scores = ms.validation_curve(svm.SVC(), train_data, train_label,
                                                'gamma', gamma_range,
                                                cv=10, scoring='accuracy')


print(train_scores.shape)
print(test_scores.shape)

train_scores = train_scores.mean(axis=1)
test_scores = test_scores.mean(axis=1)

plt.plot(gamma_range, train_scores, 'o-', label='train acc')
plt.plot(gamma_range, test_scores, 'o-', label='test acc')
plt.xlabel('gamma')
plt.ylabel('acc')
plt.legend(loc='best')
plt.show()

# 选择好参数之后就可以在整个训练集上进行训练,并在测试集上进行测试
clf = svm.SVC(gamma=9e-4)
clf.fit(train_data, train_label)
pred_label = clf.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0]) * 100)
