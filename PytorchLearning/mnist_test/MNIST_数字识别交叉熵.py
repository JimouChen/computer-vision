"""
# @Time    :  2020/10/21
# @Author  :  Jimou Chen
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 结构中只有输入和输出层
        self.fc1 = nn.Linear(784, 10)
        # 给一个激活函数,dim=1是第一个维度，即输出第一个维度的概率
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # 全连接层把(64, 1, 28, 28)转换为二维(64, 784),view相当于reshape,784=1*28*28
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        x = self.softmax(x)
        return x


if __name__ == '__main__':
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
    # 装载训练集，shuffle=True将数据打乱
    train_load = DataLoader(dataset=train_data,
                            batch_size=batch_size,
                            shuffle=True)
    # 装载测试集，将数据打乱
    test_load = DataLoader(dataset=test_data,
                           batch_size=batch_size,
                           shuffle=True)

    # for i, data in enumerate(train_load):
    #     inputs, labels = data
    #     print(inputs.shape)
    #     print(labels.shape)
    #     print(labels)
    #     break

    # 定义模型,损失函数，优化器
    model = Net()
    cross_loss = nn.CrossEntropyLoss()
    opt = optim.SGD(model.parameters(), lr=0.5)


    def train():
        for i, data in enumerate(train_load):
            # 每一次迭代都返回一组输入数据和标签
            input_data, labels = data
            # 获得模型的结果
            out = model(input_data)
            # 使用交叉熵代价函数，shape不用一样，out(batch, class_num),labels(batch)
            loss = cross_loss(out, labels)
            # 梯度清零
            opt.zero_grad()
            # 计算梯度
            loss.backward()
            # 修改权值
            opt.step()


    # 定义一个测试数据的函数
    def test():
        correct = 0
        for i, data in enumerate(test_load):
            # 每一次迭代都返回一组输入数据和标签
            input_data, labels = data
            # 获得模型的结果
            out = model(input_data)
            # 获得第一个维度的最大值，以及最大值所在的位置
            max_value, pred_index = torch.max(out, 1)
            # 用这64个预测数据与标签做一个对比，统计预测正确的数量
            correct += (pred_index == labels).sum()

        print('准确率：{0}'.format(correct.item() / len(test_data)))


    # 训练和测试10个周期
    for i in range(10):
        print(i, ':', end='')
        train()
        test()
