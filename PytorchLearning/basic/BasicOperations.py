"""
# @Time    :  2020/9/14
# @Author  :  Jimou Chen
"""
import torch

a = torch.randint(1, 10, (2, 3))
b = torch.randint(1, 10, (2, 3))
print(a, b)

# 对应位置的运算
print(a + b)
print(a - b)
print(a * b)
print(a // b)
print(a % b)

# 类似矩阵的乘法
t = torch.ones(3, 5)
# t是float的类型，需要两者类型一致
a = a.float()
print(torch.matmul(a, t))
# 转置
print(a.T)
