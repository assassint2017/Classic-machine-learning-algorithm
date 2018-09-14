"""

测试脚本
"""

import csv

import torch
from torch.utils.data import DataLoader

from LeNet import LeNet
from dataset import test_ds


# Hyper parameters
batch_size = 128
workers = 2  # subprocess number for load the image
module_dir = './modle/net299-329.pth'

pred_label = []

# dataset
test_dl = DataLoader(test_ds, batch_size, num_workers=workers)

# use cuda if you have GPU
net = LeNet().cuda()
net.load_state_dict(torch.load(module_dir))
net.eval()

# 预测结果
for step, data in enumerate(test_dl, 1):

    data = data.cuda()

    with torch.no_grad():
        outputs = net(data)

    outputs = torch.max(outputs, 1)[1].data.cpu().numpy().tolist()
    pred_label += outputs


# 将预测结果写入到csv文件中
with open('pred.csv', 'w') as file:
    writer = csv.writer(file)

    writer.writerow(['ImageId', 'Label'])

    for index, label in enumerate(pred_label, start=1):
        line = [str(index), str(label)]
        writer.writerow(line)
