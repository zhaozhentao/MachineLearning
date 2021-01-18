import numpy as np
import tensorflow as tf

import dataset as d

# step1 产生模拟数据
training_x = d.training_x
training_y = d.training_y

# step2 构建网络
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=training_x[0].shape, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy')

# step3 训练
model.fit(training_x, training_y, epochs=3000)

# 预测
class1_sample = [1, 2]
class2_sample = [35, 38]
y = model.predict(np.array([class1_sample, class2_sample]))

for i in np.nditer(y):
    if i < 0.5:
        print('class1')
    elif i > 0.5:
        print('class2')
