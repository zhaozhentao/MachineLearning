import matplotlib.pyplot as plt
import numpy as np


def _sigmoid(z):
    return np.clip(1 / (1.0 + np.exp(-z)), 1e-8, 1 - (1e-8))


def _f(X, w, b):
    return _sigmoid(np.matmul(X, w) + b)


def _gradient(X, Y_label, w, b):
    y_pred = _f(X, w, b)
    pred_error = Y_label - y_pred
    w_grad = -np.sum(pred_error * X.T, 1)
    b_grad = -np.sum(pred_error)
    return w_grad, b_grad


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

# 随机初始化weights & bias
dimension = training_x.shape[1]
w = np.zeros((dimension,))
b = np.zeros((1,))
learning_rate = 0.01

# 梯度下降
for _ in range(1):
    w_grad, b_grad = _gradient(training_x, training_y, w, b)
    pass
