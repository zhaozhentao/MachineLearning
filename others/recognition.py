import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from locate_plate import locate, create_mask

char_dict = {"京": 0, "沪": 1, "津": 2, "渝": 3, "冀": 4, "晋": 5, "蒙": 6, "辽": 7, "吉": 8, "黑": 9, "苏": 10,
             "浙": 11, "皖": 12, "闽": 13, "赣": 14, "鲁": 15, "豫": 16, "鄂": 17, "湘": 18, "粤": 19, "桂": 20,
             "琼": 21, "川": 22, "贵": 23, "云": 24, "藏": 25, "陕": 26, "甘": 27, "青": 28, "宁": 29, "新": 30,
             "0": 31, "1": 32, "2": 33, "3": 34, "4": 35, "5": 36, "6": 37, "7": 38, "8": 39, "9": 40,
             "A": 41, "B": 42, "C": 43, "D": 44, "E": 45, "F": 46, "G": 47, "H": 48, "J": 49, "K": 50,
             "L": 51, "M": 52, "N": 53, "P": 54, "Q": 55, "R": 56, "S": 57, "T": 58, "U": 59, "V": 60,
             "W": 61, "X": 62, "Y": 63, "Z": 64}

index_to_char = [
    "京", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏",
    "浙", "皖", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼",
    "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新",
    "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
    "A", "B", "C", "D", "E", "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z"
]


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
    plate_chars = recognition_model.predict(np.array([plate]))
    plate = []
    for cs in plate_chars:
        plate.append(index_to_char[np.argmax(cs)])
    print(''.join(plate))
    break

plt.show()
