import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# 随机创建100个x轴变量, 模拟生成有线性关系的数据
x = np.random.random(50)
# 加入随机的噪声
noise = np.random.rand(50, ) / 10
# f(x) = 2x + 1
y = 2 * x + 1 + noise

# 显示x, y散点图
plt.scatter(x, y)

model = tf.keras.Sequential([
    # 网络输出是1个value y, 输入是1个value x (y = wx + b)
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='adam', loss='mse')
# 查看网络参数
model.summary()
# 开始训练
model.fit(x, y, epochs=3000)

# 用训练完的模型来预测
valid_x = [0.9, 0.8, 0.7]
predict_y = model.predict(valid_x)

print(predict_y)

plt.scatter(valid_x, predict_y)
plt.show()
