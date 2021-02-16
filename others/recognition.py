import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from locate_plate import locate, create_mask


def load_image(path):
    image = tf.io.read_file(path + '/img.png')
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [width, height])
    image /= 255.0
    return image


width = 416
height = 416
images_path = [str(p) for p in pathlib.Path('./dataset').glob('*/*')]
detect_model = tf.keras.models.load_model('zc.h5')
recognition_model = tf.keras.models.load_model('plate.h5')

np.random.shuffle(images_path)

for p in images_path:
    img = tf.io.read_file(p + '/img.png')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [width, height])
    img = img / 255.0

    result = detect_model.predict(np.array([img]))

    mask = create_mask(result)
    mask = tf.keras.preprocessing.image.array_to_img(mask)
    mask = np.asarray(mask)
    img = np.asarray(img)

    plate = locate(img, mask)
    plt.imshow(plate)
    break

plt.show()
