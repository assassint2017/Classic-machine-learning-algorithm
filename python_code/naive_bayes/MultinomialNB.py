"""

朴素贝叶斯分类器中的多项式模式
用于处理离散特征
"""

from sklearn import naive_bayes
from sklearn import datasets
import sklearn.model_selection as ms

bayes = naive_bayes.MultinomialNB()

digits = datasets.load_digits()
data = digits.data
label = digits.target

train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=1)

bayes.fit(train_data, train_label)

pred_label = bayes.predict(test_data)
print('test acc:{:.3f}%'.format(sum(pred_label == test_label) / test_label.shape[0] * 100))

# test acc:90.185%
