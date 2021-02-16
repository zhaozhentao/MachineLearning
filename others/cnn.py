import pathlib

import numpy as np
import tensorflow as tf

from common import char_dict

# 读取数据集
all_image_path = [str(p) for p in pathlib.Path('./dataset').glob('*/*')]

n = len(all_image_path)
X_train, y_train = [], []
for i in range(n):
    path = all_image_path[i]
    print('正在读取 {}'.format(all_image_path[i]))
    img = tf.io.read_file(path + '/plate.jpeg')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [80, 240])
    img /= 255.0
    plate = pathlib.Path(path).name
    label = [char_dict[name] for name in plate[0:7]]  # 图片名前7位为车牌标签
    X_train.append(img)
    y_train.append(label)

X_train = np.array(X_train)
y_train = [np.array(y_train)[:, i] for i in range(7)]

input = tf.keras.layers.Input((80, 240, 3))
x = input
x = tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), strides=1, padding='same', activation='relu')(x)
x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
for i in range(3):
    x = tf.keras.layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = tf.keras.layers.Conv2D(filters=32 * 2 ** i, kernel_size=(3, 3), padding='valid', activation='relu')(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2), padding='same', strides=2)(x)
    x = tf.keras.layers.Dropout(0.5)(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dropout(0.3)(x)
Output = [tf.keras.layers.Dense(65, activation='softmax', name='c%d' % (i + 1))(x) for i in range(7)]

model = tf.keras.models.Model(inputs=input, outputs=Output)

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, y_train, epochs=100)
