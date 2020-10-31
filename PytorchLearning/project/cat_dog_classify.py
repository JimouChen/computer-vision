"""
# @Time    :  2020/10/31
# @Author  :  Jimou Chen
"""
import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import sys

if __name__ == '__main__':
    # 数据预处理,transforms用来处理数据，用来数据增强
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),  # 对图像进行随机裁剪，然后重新调整大小为224*224
        transforms.RandomRotation(20),  # 随机选择角度，顺或者逆时针20读
        transforms.RandomHorizontalFlip(p=0.5),  # 50%的概率进行随机水平翻转
        transforms.ToTensor()
    ])

    # 读取数据
    train_dataset = datasets.ImageFolder('image/train', transform)
    test_dataset = datasets.ImageFolder('image/test', transform)

    # 导入数据,设置批次和打乱
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=8, shuffle=True)

    # print(train_dataset.classes)
    model = models.vgg16(pretrained=False)
    print(model)

    # 加这句话，不训练卷积层，只训练模型的全连接层，不更新权值
    # 因为他已经把卷积层训练得非常好了，我们可以不用去训练它
    # 如果没加这句话，那准确率会更高，但是训练时间更长
    for p in model.parameters():
        p.requires_grad = False
    # 构建新的全连接层,25088和上面model的输入一致，输出是自定义的2个类
    model.classifier = nn.Sequential(
        nn.Linear(25088, 100),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(100, 2)
    )

    # 定义代价函数
    entropy_loss = nn.CrossEntropyLoss()
    # 定义优化器
    opt = optim.Adam(model.parameters(), lr=0.0003)


    def train():
        model.train()
        for i, data in enumerate(train_loader):
            # 获取数据和对应标签
            inputs, labels = data
            # 获得预测结果
            out = model(inputs)
            # 代价函数
            loss = entropy_loss(out, labels)
            # 梯度清零
            loss.backward()
            # 修改权值
            opt.step()


    def test():
        model.eval()
        correct = 0
        for i, data in enumerate(test_loader):
            # 获得数据和对应的标签
            inputs, labels = data
            # 获得模型预测结果
            out = model(inputs)
            # 获得最大值，以及最大值所在的位置
            _, predicted = torch.max(out, 1)
            # 预测正确的数量
            correct += (predicted == labels).sum()
        print("Test acc: {0}".format(correct.item() / len(test_dataset)))

        correct = 0
        for i, data in enumerate(train_loader):
            # 获得数据和对应的标签
            inputs, labels = data
            # 获得模型预测结果
            out = model(inputs)
            # 获得最大值，以及最大值所在的位置
            _, predicted = torch.max(out, 1)
            # 预测正确的数量
            correct += (predicted == labels).sum()
        print("Train acc: {0}".format(correct.item() / len(train_dataset)))


    for epoch in range(10):
        print('周期:', epoch)
        train()
        test()

    # 保存该模型
    torch.save(model.state_dict(), 'cat_dog.pth')
