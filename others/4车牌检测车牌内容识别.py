import pathlib

import numpy as np
import tensorflow as tf

from common import locate, create_mask, index_to_char


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

error_count = 0
idx = 0
for p in images_path:
    print('idx {}'.format(idx))
    idx += 1
    img = tf.io.read_file(p + '/img.png')
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [width, height])
    raw_img = img
    img = img / 255.0

    result = detect_model.predict(np.array([img]))

    mask = create_mask(result)
    mask = tf.keras.preprocessing.image.array_to_img(mask)
    mask = np.asarray(mask)
    img = np.asarray(img)

    plate_image = locate(img, mask)
    plate_chars = recognition_model.predict(np.array([plate_image]))
    plate = []
    for cs in plate_chars:
        plate.append(index_to_char[np.argmax(cs)])

    real_plate = pathlib.Path(p).name
    predict_plate = ''.join(plate)

    if predict_plate != real_plate:
        print('wrong real plate is {}, predict plate is {}'.format(real_plate, predict_plate))
        error_count += 1
        raw_img = np.asarray(raw_img)
        plate_image = locate(raw_img, mask)
        tf.io.write_file('./dataset/error/' + real_plate + '/plate.jpeg', tf.image.encode_jpeg(plate_image))

print('total error {}'.format(error_count))
