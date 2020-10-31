"""
# @Time    :  2020/10/27
# @Author  :  Jimou Chen
"""
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class LSTM_Net(nn.Module):
    def __init__(self):
        super(LSTM_Net, self).__init__()
        # 输入特征的大小，LSTM模块的数量，LSTM的层数
        self.lstm = nn.LSTM(input_size=28, hidden_size=64, num_layers=1, batch_first=True)

        # 设置隐藏层,64个输入，10个输出的全连接
        self.out = nn.Linear(64, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        # LSTM要求输入数据必须是三维,-1是自动匹配
        # (批次64，序列长度28， 特征数量28)
        x = x.view(-1, 28, 28)
        output, (h_n, c_n) = self.lstm(x)
        output_in_last_timestep = h_n[-1, :, :]
        x = self.out(output_in_last_timestep)
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

    # 定义模型,损失函数，优化器
    model = LSTM_Net()
    cross_loss = nn.CrossEntropyLoss()
    opt = optim.Adam(model.parameters(), lr=0.001)  # 使用Adam时，把学习率改小一点


    def train():
        # 设为训练状态，使用训练时dropout起作用
        model.train()
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
        # 设为测试状态，使用训练时dropout不起作用
        model.eval()
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

        print('测试集准确率：{0}'.format(correct.item() / len(test_data)))

        # 下面对训练数据进行测试
        correct = 0
        for i, data in enumerate(train_load):
            # 每一次迭代都返回一组输入数据和标签
            input_data, labels = data
            # 获得模型的结果
            out = model(input_data)
            # 获得第一个维度的最大值，以及最大值所在的位置
            max_value, pred_index = torch.max(out, 1)
            # 用这64个预测数据与标签做一个对比，统计预测正确的数量
            correct += (pred_index == labels).sum()

        print('训练集准确率：{0}'.format(correct.item() / len(train_data)))


    # 训练和测试20个周期
    for i in range(20):
        print(i, ':', end='')
        train()
        test()
