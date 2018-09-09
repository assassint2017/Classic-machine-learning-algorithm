"""

使用线性回归预测boston的房价
"""

from sklearn import datasets
import sklearn.model_selection as ms
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing

# 获取数据
boston = datasets.load_boston()
data = boston.data
label = boston.target
"""
用于回归预测boston房价的数据集
共506例,13种特征
"""
# print(data)
# print(label)
# print('---------')
# print(data.shape)
# print(label.shape)

# 对数据进行缩放
data = preprocessing.scale(data)  # 这个函数就是减去均值除以std,其中std的计算方式就是正常的那种

# 拆分数据为训练集和测试集,最后一个参数就是随机种子点
train_data, test_data, train_label, test_label = ms.train_test_split(data, label, test_size=0.3, random_state=4)

# 建立一个线性回归的模型
lr = LinearRegression()
lr.fit(train_data, train_label)

# 在测试集上进行测试
pred_label = lr.predict(test_data)

# 进行评价指标(MSE)的计算和打印
print('MSE:{:.3f}'.format(((pred_label - test_label) ** 2).mean()))
print(lr.score(test_data, test_label))  # 按照一定的规则,对模型进行评价,这里使用的是一个叫coefficient of determination R^2的指标

# 打印模型参数
print('-----------')
print(lr.coef_)  # 对应着weight
print(lr.intercept_)  # intercept 截距,对应着bias
print(lr.get_params())  # 用于打印对于模型的设定,可以理解为超参数

