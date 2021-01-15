import matplotlib.pyplot as plt
import numpy as np


# 计算梯度
def compute_gradient(current_w, current_b, training_x, training_y):
    N = float(len(training_x))
    w_gradient = 2. / N * ((current_w * training_x + current_b - training_y) * training_x).sum()
    b_gradient = 2. / N * (current_w * training_x + current_b - training_y).sum()
    return w_gradient, b_gradient


# step1 产生模拟数据 y = 2x + 30
train_x = np.random.random(50)
train_y = 3 * train_x + 15

# step2 随机选取w,b初始值
w = 0
b = 0
learning_rate = 0.01

# step3 使用梯度下降法寻找最佳参数
for i in range(6000):
    w_gradient, b_gradient = compute_gradient(w, b, train_x, train_y)
    # 梯度下降更新w, b
    w = w - learning_rate * w_gradient
    b = b - learning_rate * b_gradient
    print("w:{}  b:{}".format(w, b))

plt.scatter(train_x, train_y)
plt.show()
