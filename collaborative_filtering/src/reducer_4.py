#!/usr/bin/python3

import sys
import numpy as np

def compute_rating(sims, ratings):
    sims_sum = np.sum(sims) + 1e-9
    return np.dot(sims, ratings) / sims_sum

skip_item = False
prev_key = None
sims = []
ratings = []
for line in sys.stdin:
    line = line.strip()
    key, val = line.split('\t')
    if prev_key == None:
        prev_key = key
    if key != prev_key:
        if not skip_item:
            r_pred = compute_rating(sims, ratings)
            u, i = prev_key.split('!')
            print('{}\t{:.3f}\t{}'.format(u, r_pred, i))
        prev_key = key
        skip_item = False
        sims = []
        ratings = []
    if val == '_':
        skip_item = True
    if skip_item:
        continue
    sim, r = val.split('!')
    sims.append(float(sim))
    ratings.append(float(r))

r_pred = compute_rating(sims, ratings)
u, i = key.split('!')
print('{}\t{:.3f}\t{}'.format(u, r_pred, i))
