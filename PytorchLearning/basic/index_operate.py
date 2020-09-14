"""
# @Time    :  2020/9/14
# @Author  :  Jimou Chen
"""
import torch

a = torch.arange(1, 13)
print(a)

for i in a:
    print(i)
    print(i.item())

print(a[3:8])

index = [1, 5, 7]
print(a[index])
