#!/usr/bin/python3

import sys
import numpy as np

for line in sys.stdin:
    line = line.strip()
    u, val = line.split('\t')
    items, ratings = val.split('!')
    item_lst = items.split(',')
    ratings = np.array(list(map(float, ratings.split(','))))
    ratings -= ratings.mean()
    ratings = ratings.round(3).astype(str)
    val_centered = '{}!{}'.format(items, ','.join(ratings))
    for i, r in zip(item_lst, ratings):
        print('{}\t{}!{}'.format(i, r, val_centered))
