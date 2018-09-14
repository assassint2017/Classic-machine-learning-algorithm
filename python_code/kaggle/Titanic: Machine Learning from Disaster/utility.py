"""

数据说明,一例数据共12维属性

PassengerId 乘客的ID
Survived 是否幸存 0 = No, 1 = Yes

Pclass 船票的等级 共分为123 三个等级
Name 年龄
Sex 性别
Age 有少量缺失值
SibSp 兄弟姐妹+配偶的数量
Parch 父母+子女的数量
Ticket 船票的号码
Fare 乘客的票价(每个人都不太一样,有贵的,也有便宜的)
Cabin 船舱的编号(有大量缺失值)
Embarked 登岸港口(有三个不同的值)

其中很明显 船票的编号 明显和幸存是否没有关系

其中虽然大部分离散特征为有序的,只有Embarked为无序的,因此要处理为独热码,剩下的直接处理就行

训练集中
总人数:891
幸存人数:342
遇难人数:549
"""

import pandas as pd


def get_data(training):
    """

    :param training: 是否读取训练集数据
    :return:
    """
    if training is True:
        csv = pd.read_csv('./all/train.csv')
        label = csv['Survived'].values
        del csv['Survived']
    else:
        csv = pd.read_csv('./all/test.csv')

    # 开始对数据进行处理
    def set_sex(sex):
        """

        :param sex: 定义男性为0 ,女性为1
        """
        if 'female' in sex:
            return 1
        elif 'male' in sex:
            return 0

    csv['Sex'] = csv['Sex'].apply(set_sex)

    csv['child'] = csv['Age'].apply(lambda age: 0 if age > 15 else 1)

    csv['family_size'] = csv['SibSp'] + csv['Parch'] + 1

    def set_embarked_c(embarked):
        if embarked is 'C':
            return 1
        else:
            return 0
    csv['Embarked_C'] = csv['Embarked'].apply(set_embarked_c)

    def set_embarked_s(embarked):
        if embarked is 'S':
            return 1
        else:
            return 0
    csv['Embarked_S'] = csv['Embarked'].apply(set_embarked_s)

    def set_embarked_q(embarked):
        if embarked is 'Q':
            return 1
        else:
            return 0
    csv['Embarked_Q'] = csv['Embarked'].apply(set_embarked_q)

    del csv['Age']
    del csv['Ticket']
    del csv['Cabin']
    del csv['Name']
    del csv['PassengerId']
    del csv['Embarked']
    del csv['SibSp']
    del csv['Parch']

    data = csv.values

    # 返回结果
    if training is True:
        return data, label
    else:
        return data
