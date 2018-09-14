"""

训练脚本
"""

from time import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from LeNet import LeNet
from dataset import train_ds


# Hyper parameters
epoch_num = 300
batch_size = 128
lr = 1e-4  # learning rate
workers = 2  # subprocess number for load the image
weight_decay = 1e-3

train_ds_size = 42000  # the size of your train dataset


# dataset
train_dl = DataLoader(train_ds, batch_size, True, num_workers=workers)

# use cuda if you have GPU
net = LeNet()
net = net.cuda()

# optimizer
opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)  # optimizer for network

# loss function
loss_func = nn.CrossEntropyLoss()

# train the network
start = time()

for epoch in range(epoch_num):

    for step, (data, target) in enumerate(train_dl, 1):

        data, target = data.cuda(), target.cuda()

        outputs = net(data)

        loss = loss_func(outputs, target)

        opt.zero_grad()

        loss.backward()

        opt.step()

    # 每个epoch进行测试一下精度
    net.eval()

    train_acc = 0

    for test_step, (data, target) in enumerate(train_dl, 1):

        data, target = data.cuda(), target.cuda()

        outputs = net(data)

        train_acc += sum(torch.max(outputs, 1)[1].data.cpu().numpy() == target.data.cpu().numpy())

    train_acc /= train_ds_size

    net.train()

    print('epoch:{}, train_acc:{:.3f} %, loss:{:.3f}, time:{:.1f} min'
          .format(epoch, train_acc * 100, loss.data.item(), (time() - start) / 60))

torch.save(net.state_dict(), './modle/net{}-{}.pth'.format(epoch, step))
