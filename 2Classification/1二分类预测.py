import numpy as np

import dataset as d


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


# step1 产生模拟数据
training_x = d.training_x
training_y = d.training_y

# step2 随机初始化weights & bias
dimension = training_x.shape[1]
w = np.zeros((dimension,))
b = np.zeros((1,))
learning_rate = 0.01
step = 1

# step3 梯度下降训练
for _ in range(6000):
    w_grad, b_grad = _gradient(training_x, training_y, w, b)
    w = w - learning_rate / np.sqrt(step) * w_grad
    b = b - learning_rate / np.sqrt(step) * b_grad
    step = step + 1

# 预测
# should be close to 0 belong class1
print(_f(np.array([[5, 5]]), w, b))

# should be close to 1 belong class2
print(_f(np.array([[25, 25]]), w, b))
