import pathlib

import matplotlib.pyplot as plt
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


def load_and_preprocess_image(path):
    image = tf.io.read_file(path + '/img.png')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [192, 192])
    image /= 255.0

    mask = tf.io.read_file(path + '/label.png')
    mask = tf.image.decode_jpeg(mask, channels=3)
    mask = tf.image.resize(mask, [192, 192])
    mask /= 255.0
    return image, mask


data_root = pathlib.Path('/Users/zhaotao/Desktop/py/4ObjectDetect/data')

all_image_paths = list(data_root.glob('*'))
all_image_paths = [str(path) for path in all_image_paths]

path_ds = tf.data.Dataset.from_tensor_slices(all_image_paths)
image_ds = path_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)

for image, mask in image_ds.take(1):
    plt.imshow(mask)

plt.show()
