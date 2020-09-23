"""
# @Time    :  2020/9/15
# @Author  :  Jimou Chen
"""
import numpy as np
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# 训练集
train_data = datasets.MNIST(root='./',
                            train=True,
                            transform=transforms.ToTensor(),
                            download=True)
# 测试集
test_data = datasets.MNIST(root='./',
                           train=False,
                           transform=transforms.ToTensor(),
                           download=True)

# 批次大小，即一次加载多少数据
batch_size = 64
# 装载训练集，将数据打乱
train_load = DataLoader(dataset=train_data,
                        batch_size=batch_size,
                        shuffle=True)
# 装载测试集，将数据打乱
test_load = DataLoader(dataset=test_data,
                       batch_size=batch_size,
                       shuffle=True)
