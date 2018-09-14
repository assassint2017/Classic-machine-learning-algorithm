import csv

import numpy as np


def get_train_data():
    """

    :return: 获取训练数据
    """
    train_data_path = './all/train.csv'
    train_list = []
    label = []

    # 从csv文件中读取数据
    with open(train_data_path, 'r') as file:
        lines = csv.reader(file)
        for index, line in enumerate(lines):
            if index is not 0:
                label.append(int(line[0]))
                train_list.append(list(map(int, line[1:])))

    # 将图像数据转换为ndarray
    train_data = np.array(train_list).reshape((42000, 1, 28, 28))

    # 将标签数据转换成ndarray
    label = np.array(label)

    return train_data, label


def get_test_data():
    """

    :return: 获取测试数据
    """
    test_data_path = './all/test.csv'
    test_list = []

    # 从csv文件中读取数据
    with open(test_data_path, 'r') as file:
        lines = csv.reader(file)
        for index, line in enumerate(lines):
            if index is not 0:
                test_list.append(list(map(int, line)))

    # 将图像数据转换为ndarray
    test_data = np.array(test_list).reshape((28000, 1, 28, 28))

    return test_data
