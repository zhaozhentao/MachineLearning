import matplotlib.pyplot as plt
import tensorflow as tf

# 载入并准备好 MNIST 数据集。将样本从整数转换为浮点数
mnist = tf.keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# 显示一张样本图片
plt.imshow(x_train[0])
plt.show()

model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir='./log')

if False:
    # 顺序标记的label，使用sparse_categorical_crossentropy作为损失函数
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, y_train, epochs=5, callbacks=[tensorboard])

    # 验证模型
    model.evaluate(x_test, y_test, verbose=2)
else:
    train_y_one_hot_label = tf.keras.utils.to_categorical(y_train)
    test_y_one_hot_label = tf.keras.utils.to_categorical(y_test)

    # 独热编码label，使用categorical_crossentropy作为损失函数
    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.fit(x_train, train_y_one_hot_label, epochs=5, callbacks=[tensorboard])
    # 验证模型
    model.evaluate(x_test, test_y_one_hot_label, verbose=2)
