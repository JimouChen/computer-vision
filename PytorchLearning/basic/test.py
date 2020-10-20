# """
# # @Time    :  2020/9/14
# # @Author  :  Jimou Chen
# """
# import torch
#
# b = torch.tensor([1, 2, 3], dtype=float)
# a = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]])
# print('维度：', a.ndim)
# print('形状：', a.shape, a.size())
#
# a = torch.ones([2, 3])
# b = torch.zeros([4, 3])
# print(a)
# print(b)
#
# # 生成0-1之间随机数
# print(torch.rand(3, 3))
# # 0-10随机整数
# print(torch.randint(0, 10, (2, 3)))
# # 标准正态分布的随机数
# print(torch.randn(2, 2))
#
# a = torch.tensor([[1, 2, 3],
#                   [4, 5, 6]])
# # 生成一个维度和a一样的数据,类型是float
# b = torch.rand_like(a, dtype=float)
# print(b)
# b = b.reshape(6)
# print(b)
# # 取里面的值
# print(b[1].item())
# b = b.reshape(3, 2)
# print(b)
# print(b[0][1].item())
#
# # tensor 转换成array
# import numpy as np
#
# a = np.array(b)
# print(a)
# # array 转换成tensor
# a = torch.tensor(a)
# print(a)
import matplotlib.pyplot as plt
# a = [[1], [1]]
# b = [[2], [2]]
#
#
# plt.scatter(a, b)
# plt.show()
import numpy as np
if __name__ == '__main__':
    # 随机生成0-1内的100个数据
    x_data = np.random.rand(5)
    print(x_data)
    # 0-0.01内的噪点数据，尺寸和x_data一样
    # noise = np.random.normal(0, 0.01, x_data.shape)
    # print(noise.shape,x_data.shape)
    # 因变量
    y_data = x_data * 0.1 + 0.2
    print(y_data.shape)

    plt.scatter(x_data, y_data)
    plt.show()