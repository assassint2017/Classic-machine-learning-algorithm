import matplotlib.pyplot as plt

import sklearn.tree as tree
import sklearn.model_selection as ms
from sklearn import datasets

# 获取数据
dataset = datasets.make_regression(10, 1, noise=10)
# data, label = dataset.data, datasets.target
print(len(dataset))
print(dataset[0].shape)
print(dataset[1].shape)

train_data, test_data, train_label, test_label = ms.train_test_split(dataset[0], dataset[1], test_size=0.3, random_state=3)

# 构建模型
reg = tree.DecisionTreeRegressor(max_depth=3)
reg.fit(train_data, train_label)

pred_label = reg.predict(test_data)

plt.scatter(test_data, test_label, label='GT')
plt.scatter(test_data, pred_label, label='pred')
plt.legend(loc='best')
plt.show()

"""
graphviz 是一个由AT&T实验室启动的开源工具包，用于绘制DOT语言脚本描述的图形
DOT语言是一种文本图形描述语言。它提供了一种简单的描述图形的方法，并且可以为人类和计算机程序所理解。DOT语言文件通常是具有.gv或.dot的文件扩展名

从已有的dot文件导出命令
＜cmd＞ dot -T ＜format＞ ＜inputfile＞ -o ＜outputfile＞

例如:
dot -T pdf tree.dot -o tree.pdf

具体可支持的格式,可以查阅官网链接
https://graphviz.gitlab.io/_pages/doc/info/output.html

常用的一些输出格式比如png和pdf,其中pdf是可以直接复制上边的文字的,而不是简单的图片贴在上边
这点非常的棒
"""

# 通过图,验证了之前对于回归树的想法,就是不断的去二分数据
# rounded是使用圆角矩形来显示,filled是用颜色去填充矩形
tree.export_graphviz(reg, 'tree.dot', rounded=True, filled=True)
