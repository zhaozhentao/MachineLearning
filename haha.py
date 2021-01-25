import os
import numpy as np
import shutil

free = '/Users/zhaotao/Desktop/parkinglot/train/occupy'
validate_free = '/Users/zhaotao/Desktop/parkinglot/validate/occupy'

files = os.listdir(free)

fileIndex = np.arange(len(files))
np.random.shuffle(fileIndex)
pre10 = fileIndex[:int(len(fileIndex) / 10)]

for i, index in enumerate(pre10):
    file = os.path.join(free, files[index])
    print('move ' + file)
    shutil.move(file, validate_free)

print('finish')
