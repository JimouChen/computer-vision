### 独热编码说明
```python
# 转换为独热编码，input_data.size[0]是64
# dim=1, index=labels, src=1
one_hot = torch.zeros(input_data.shape[0], 10).scatter(1, labels, 1)
```
### 参数说明

- dim:维度，对哪个维度进行独热编码
- index:要将src中对应的值放在tensor中的哪个位置
- src:插入index的数值

> 实现的效果如下图所示：

![](https://img2020.cnblogs.com/blog/2134757/202010/2134757-20201020232722823-1739461755.png)

- 进行独热编码的目的是保证计算loss时数据和标签的维度一致