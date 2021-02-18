import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix


def load_and_preprocess_image(path):
    img = tf.io.read_file(path + '/img.png')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [128, 128])
    img /= 255.0

    label = tf.io.read_file(path + '/label.png')
    label = tf.image.decode_jpeg(label, channels=3)
    label = tf.image.resize(label, [128, 128])
    # 3 通道降为 1 通道
    label = tf.image.rgb_to_grayscale(label)
    label /= 38.0

    return img, label


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[128, 128, 3])
    x = inputs

    # 在模型中降频取样
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # 升频取样然后建立跳跃连接
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # 这是模型的最后一层
    last = tf.keras.layers.Conv2DTranspose(
        output_channels, 3, strides=2,
        padding='same')  # 64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=2):
    if dataset:
        for img, mask in dataset.take(num):
            pred_mask = model.predict(img)
            display([img[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])


batch_size = 64
output_channels = 1
all_image_paths = [str(path) for path in pathlib.Path('./data').glob('*')]
image_count = len(all_image_paths)
steps_per_epoch = tf.math.ceil(image_count / batch_size).numpy()

image_ds = (
    tf.data.Dataset.from_tensor_slices(all_image_paths)
        .map(load_and_preprocess_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
)

train_dataset = (
    image_ds.cache()
        .shuffle(image_count)
        .repeat()
        .batch(batch_size)
        .prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
)

for image, mask in image_ds.take(2):
    sample_image, sample_mask = image, mask

display([sample_image, sample_mask])


base_model = tf.keras.applications.MobileNetV2(input_shape=[128, 128, 3], include_top=False)

# 使用这些层的激活设置
layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',  # 4x4
]
layers = [base_model.get_layer(name).output for name in layer_names]

# 创建特征提取模型
down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)

down_stack.trainable = False

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),  # 32x32 -> 64x64
]

model = unet_model(output_channels)

model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

show_predictions()

model_history = model.fit(train_dataset, epochs=30, steps_per_epoch=steps_per_epoch)

show_predictions(train_dataset)
