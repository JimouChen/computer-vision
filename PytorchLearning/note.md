## 交叉熵
- 一般情况下，回归使用二次代价函数， 分类使用交叉熵代价函数
- 使用交叉熵代价函数不需要独热编码，也就是out和label的shape不用一致
- 交叉熵会使得模型的训练速度更加快

```python
cross_loss = nn.CrossEntropyLoss()
```

```python
input_data, labels = data
# 获得模型的结果
out = model(input_data)
# 使用交叉熵代价函数，shape不用一样，out(batch, class_num),labels(batch)
loss = cross_loss(out, labels)
```

- res
```python
0 :准确率：0.9037
1 :准确率：0.913
2 :准确率：0.9171
3 :准确率：0.9186
4 :准确率：0.9193
5 :准确率：0.921
6 :准确率：0.9218
7 :准确率：0.9244
8 :准确率：0.9244
9 :准确率：0.9234

Process finished with exit code 0

```

## dropout
- 有时候用了效果不一定比没有用好
- 在数据量很大时，用dropout效果可能更好
- dropout可以对抗过拟合

## 设置正则化
- l2
```python
# weight_decay是l2正则化系数，加上就是设置l2正则化
opt = optim.SGD(model.parameters(), lr=0.5, weight_decay=0.0001)
```
- 可以对抗过拟合，看情况使用

## 优化器
- 所有优化器的本质都是梯度下降法
- SGD是随机梯度下降法
- 用的较多是Adam，不知道用什么优化器的时候，一般用Adam
- Adam的学习率设置的比较小一点，0.001之类的

## 池化pooling
- 最大池化(用的最多)
    - 相当于取重要度(权重)最大的那个数
- 平均池化
- 随机池化(随便从几个数据中取一个)

### 池化的作用
- 进一步提取特征
- 平移不变性

## CNN网络结构
eg:
```python
self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2))
```
- nn.Conv2d
```python
in_channels: int,输入通道，黑白取1，彩色取2
out_channels: int,特征图
kernel_size: _size_2_t,卷积数量，取3，5，7等，即3x3...
stride: _size_2_t = 1,步长，一般取1
padding: _size_2_t = 0,填充圈数，如果卷积数是3，就取1；5就取2；7就取3...
```

- nn.MaxPool2d(2, 2)
    - 参数分别是kernel_size，stride
    - 即卷积数量和步长
    - 其他参数默认

```python
# 卷积->池化->卷积->池化->全连接->全连接
self.conv1 = nn.Sequential(nn.Conv2d(1, 32, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2))
self.conv2 = nn.Sequential(nn.Conv2d(32, 64, 5, 1, 2), nn.ReLU(), nn.MaxPool2d(2, 2))
self.fc1 = nn.Sequential(nn.Linear(64*7*7, 1000), nn.Dropout(p=0.5), nn.ReLU())
self.fc2 = nn.Sequential(nn.Linear(1000, 10), nn.Softmax(dim=1))

```

- 64\*7*7 的7是因为原来是28\*28,经过两个池化层，每个池化层是原来是1/2，
所以，是28/2/2 = 7

- 注意
    - 全连接层传入的是2维的
    > 全连接层把(64, 1, 28, 28)转换为二维(64, 784),view相当于reshape,784=1*28*28
    - 卷积层要求的是4维的，如[64,1,28,28]
    > 64 是每次加载的数据量，1 是通道数，28x28是图片大小
                                                                                                                            >
                    

## RNN
- CNN一般用来处理网格化数据的神经网路，如图像数据
- RNN一般处理序列化数据的神经网络，如一段语音、一段文字、一段录像等
    - 也可以做文本分类
    - 文字翻译
    - 分词的标注
    - 通过训练之前的信息来决策当前的问题
        > 如果句子太长(序列太长)的话，可能就得到的结果会不准，因为梯度消失


## LTSM
- 要求的传入的input是三维的数据， 即
    - input(seq_len, batch, feather)
    - 序列的长度，批次的大小，特征的数量
    - 处理图像问题的话，效果比不上CNN
    
    
## nn.models
- 里面有很多已经使用imageNet训练好的模型，比如AlexNet，GoogleNet，VGG4...
- 一般最后全连接层应该加个Softmax函数转换成概率输出，但是如果代价函数使用了交叉熵的话，
也就是CrossEntropyLoss，它里面已经包含了Softmax，所以模型最后可以不用加上Softmax，
加上的话，相当于做了两次Softmax
