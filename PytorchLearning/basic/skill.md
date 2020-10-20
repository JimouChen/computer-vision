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
