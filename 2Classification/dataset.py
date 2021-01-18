import numpy as np
import matplotlib.pyplot as plt

# 模拟分类1
x11 = np.random.random_sample(100) * 10
x12 = np.random.random_sample(100) * 10
class1 = np.stack([x11, x12], axis=1)
label1 = np.zeros((100,))

# 模拟分类2
x21 = np.random.random_sample(100) * 10 + 30
x22 = np.random.random_sample(100) * 10 + 30
class2 = np.stack([x21, x22], axis=1)
label2 = np.ones((100,))

plt.scatter(x11, x12)
plt.scatter(x21, x22)
plt.show()

# 模拟数据集
training_x = np.append(class1, class2, axis=0)
training_y = np.append(label1, label2, axis=0)
