import torch

x = torch.ones((2, 2), requires_grad=True)
y = 3 * (x + 3) ** 2
# 对x求微分(导)
y = y.mean()
y.backward()  # backward自动求每一步的导数
# x.grad相当于y对x=1处求导，为6x|x=1
print(x.grad)
