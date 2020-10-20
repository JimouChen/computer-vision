## product some random data
- ####linear data
```python
import numpy as np
import matplotlib.pyplot as plt

x_data = np.random.rand(100)
noise = np.random.normal(0, 0.01, x_data.shape)
y_data = x_data * 0.1 + 0.2 + noise
plt.scatter(x_data, y_data)
plt.show()
```
![](https://img2020.cnblogs.com/blog/2134757/202010/2134757-20201020204946648-1304592291.png)


```python
# 生成-2到2间的200个数据，并设为2维
import numpy as np
import matplotlib.pyplot as plt
x_data = np.linspace(-2, 2, 200)[:, np.newaxis]
# 0-0.2内的噪点数据，尺寸和x_data一样
noise = np.random.normal(0, 0.2, x_data.shape)
# print(noise.shape,x_data.shape)
# 因变量
y_data = np.square(x_data) + noise

plt.scatter(x_data, y_data)
plt.show()

# 转成2维
x_data = x_data.reshape(-1, 1)
y_data = y_data.reshape(-1, 1)
```

![](https://img2020.cnblogs.com/blog/2134757/202010/2134757-20201020220659692-2100495190.png)
