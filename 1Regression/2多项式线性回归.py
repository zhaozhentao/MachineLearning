import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 随机创建100个x轴变量, 模拟生成有线性关系的数据
x = np.random.random(50)
# 加入随机的噪声
noise = np.random.rand(50, ) / 50
# f(x) = 2 * x * x - 2x + 1
y = 2 * x * x - 2 * x + 1 + noise

# 显示x, y散点图
plt.scatter(x, y)

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(1,), activation='tanh'),
    tf.keras.layers.Dense(2, activation='tanh'),
    tf.keras.layers.Dense(1),
])

model.compile(optimizer='adam', loss='mse')
# 开始训练
model.fit(x, y, epochs=5000)

# 用训练完的模型来预测
valid_x = np.random.random(10)
predict_y = model.predict(valid_x)

plt.scatter(valid_x, predict_y)
plt.show()
