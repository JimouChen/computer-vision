"""
# @Time    :  2020/9/15
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
import torch
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 构建神经网络模型
class UnLinearRegression(nn.Module):
    # 定义网络结构
    def __init__(self):
        super().__init__()  # 初始化父类
        '''1-10-1的结构'''
        # 加一个隐藏层，10个神经元
        self.fc1 = nn.Linear(1, 10)
        self.tanh = nn.Tanh()  # (双矩阵)激活函数
        self.fc2 = nn.Linear(10, 1)  # 全连接层/输出层

    # 定义网络计算，(前向的计算)
    def forward(self, x):
        # x 是整个网络的输入,传给全连接层做计算
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        return x  # 返回输出值


if __name__ == '__main__':
    # 生成-2到2间的200个数据，并设为2维
    x_data = np.linspace(-2, 2, 200)[:, np.newaxis]
    # 0-0.2内的噪点数据，尺寸和x_data一样
    noise = np.random.normal(0, 0.2, x_data.shape)
    # print(noise.shape,x_data.shape)
    # 因变量
    y_data = np.square(x_data) + noise

    plt.scatter(x_data, y_data)
    plt.show()

    # 转成2维
    x_data = x_data.reshape(-1, 1)
    y_data = y_data.reshape(-1, 1)

    # 把numpy类型转成tensor
    x_data = torch.FloatTensor(x_data)
    y_data = torch.FloatTensor(y_data)
    # 转换成pytorch环境里面的变量
    inputs = Variable(x_data)
    target = Variable(y_data)

    # 建模,神经网络模型
    model = UnLinearRegression()
    # 定义代价函数,取均方差代价函数
    loss = nn.MSELoss()
    # 定义优化器，取梯度下降法,传入模型参数和学习率
    optimizer = optim.SGD(model.parameters(), lr=0.3)
    # for name, param in model.named_parameters():
    #     print(name,param)

    # 训练模型1000次
    for i in range(2001):
        out = model(inputs)
        # 计算loss
        mse_loss = loss(out, target)
        # 梯度清零
        optimizer.zero_grad()
        # 计算梯度
        mse_loss.backward()
        # 修改权值
        optimizer.step()
        if i % 200 == 0:
            print(i, mse_loss.item())

    predict = model(inputs)
    plt.scatter(x_data, y_data)
    plt.plot(x_data, predict.data.numpy(), 'r-', lw=3)
    plt.show()
