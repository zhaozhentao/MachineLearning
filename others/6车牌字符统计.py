import pathlib

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['font.sans-serif'] = ['Songti SC']
glob = pathlib.Path('dataset/labeled').glob("*/*")
plates = [str(p.name) for p in glob]

chars = [[] for i in range(8)]
for plate in plates:
    for i in range(8):
        if i == 7 and len(plate) == 7:
            continue
        chars[i].append(plate[i])

character_times = [[] for i in range(8)]
for i in range(8):
    for c in np.unique(chars[i]):
        character_times[i].append({'character': c, 'count': chars[i].count(c)})

for items in character_times:
    labels = [item['character'] for item in items]
    counts = [item['count'] for item in items]
    plt.bar(range(len(counts)), counts, tick_label=labels)
    plt.show()
