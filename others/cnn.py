import pathlib

import tensorflow as tf

char_dict = {
    "京": 0.01, "沪": 0.02, "津": 0.03, "渝": 0.04, "冀": 0.05, "晋": 0.06, "蒙": 0.07, "辽": 0.08,
    "吉": 0.09, "黑": 0.10, "苏": 0.11, "浙": 0.12, "皖": 0.13, "闽": 0.14, "赣": 0.15, "鲁": 0.16,
    "豫": 0.17, "鄂": 0.18, "湘": 0.19, "粤": 0.20, "桂": 0.21, "琼": 0.22, "川": 0.23, "贵": 0.24,
    "云": 0.25, "藏": 0.26, "陕": 0.27, "甘": 0.28, "青": 0.29, "宁": 0.30, "新": 0.31,
    "0": 0.32, "1": 0.33, "2": 0.34, "3": 0.35, "4": 0.36, "5": 0.37, "6": 0.38, "7": 0.39, "8": 0.40,
    "9": 0.41, "A": 0.42, "B": 0.43, "C": 0.44, "D": 0.45, "E": 0.46, "F": 0.47, "G": 0.48, "H": 0.49,
    "J": 0.50, "K": 0.51, "L": 0.52, "M": 0.53, "N": 0.54, "P": 0.55, "Q": 0.56, "R": 0.57, "S": 0.58,
    "T": 0.59, "U": 0.60, "V": 0.61, "W": 0.62, "X": 0.63, "Y": 0.64, "Z": 0.65
}


def load_and_preprocess_image(path):
    img = tf.io.read_file(path + '/plate.jpeg')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [240, 80])
    img /= 255.0
    return img


batch_size = 32
all_image_paths = [str(path) for path in pathlib.Path('./dataset').glob('*/*')]
image_count = len(all_image_paths)

labels = []
for p in all_image_paths:
    label = [char_dict[c] for c in pathlib.Path(p).name[0:7]]
    labels.append(label)

labels = tf.data.Dataset.from_tensor_slices(labels)

l0 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[0]]
    l0.append(label)
l0 = tf.data.Dataset.from_tensor_slices(l0)

l1 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[1]]
    l1.append(label)
l1 = tf.data.Dataset.from_tensor_slices(l1)

l2 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[2]]
    l2.append(label)
l2 = tf.data.Dataset.from_tensor_slices(l2)

l3 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[3]]
    l3.append(label)
l3 = tf.data.Dataset.from_tensor_slices(l3)

l4 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[4]]
    l4.append(label)
l4 = tf.data.Dataset.from_tensor_slices(l4)

l5 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[5]]
    l5.append(label)
l5 = tf.data.Dataset.from_tensor_slices(l5)

l6 = []
for p in all_image_paths:
    label = char_dict[pathlib.Path(p).name[6]]
    l6.append(label)
l6 = tf.data.Dataset.from_tensor_slices(l6)

ds = (
    tf.data.Dataset.from_tensor_slices((all_image_paths))
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

train_ds = tf.data.Dataset.zip((ds, (l0, l1, l2, l3, l4, l5, l6)))

input = tf.keras.layers.Input((240, 80, 3))
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

model.fit(train_ds)
