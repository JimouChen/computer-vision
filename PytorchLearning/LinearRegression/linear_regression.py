"""
# @Time    :  2020/9/14
# @Author  :  Jimou Chen
"""
import numpy as np
import matplotlib.pyplot as plt
from torch import nn, optim
from torch.autograd import Variable
import torch
import os
#
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# 构建神经网络模型
class LinearRegression(nn.Module):
    # 定义网络结构
    def __init__(self):
        super().__init__()  # 初始化父类
        # 全连接层
        self.fc = nn.Linear(1, 1)  # 输入1个，输出1个

    # 定义网络计算，(前向的计算)
    def forward(self, x):
        # x 是整个网络的输入,传给全连接层做计算
        output = self.fc(x)
        return output  # 返回输出值


if __name__ == '__main__':
    # 随机生成0-1内的100个数据
    x_data = np.random.rand(100)
    # 0-0.01内的噪点数据，尺寸和x_data一样
    noise = np.random.normal(0, 0.01, x_data.shape)
    # 因变量
    y_data = x_data * 0.1 + 0.2 + noise
    print(y_data.shape)

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
    model = LinearRegression()
    # 定义代价函数,取均方差代价函数
    loss = nn.MSELoss()
    # 定义优化器，取梯度下降法,传入模型参数和学习率
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    # for name, param in model.named_parameters():
    #     print(name,param)

    # 训练模型1000次
    for i in range(1001):
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
