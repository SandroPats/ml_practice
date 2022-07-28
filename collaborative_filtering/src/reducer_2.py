#!/usr/bin/python3

import sys
import numpy as np
from collections import defaultdict

def cos_similarity(vec1, vec2):
    if len(vec1) == 1:
        return int(vec1[0] * vec2[0] > 0)
    vec_prod = np.dot(vec1, vec2)
    vec1_sqr = np.square(vec1)
    vec2_sqr = np.square(vec2)
    mask = (vec1_sqr > 0) & (vec2_sqr > 0)
    norm = np.sqrt(vec1_sqr[mask].sum() * vec2_sqr[mask].sum()) + 1e-9
    return vec_prod / norm

class SimForItem:
    def __init__(self, item_idx):
        self.item_idx = item_idx
        self.other_items = defaultdict(list) 
        self.ratings = []

    def update(self, rating, items, ratings):
        old_len = len(self.ratings)
        self.ratings.append(rating)
        item_set = set(items)
        for item in self.other_items.keys():
            if item in items:
                item_set.remove(item)
                idx = items.index(item)
                self.other_items[item].append(ratings[idx])
            else:
                self.other_items[item].append(0)
        for item in item_set:
            idx = items.index(item)
            if old_len > 0:
                self.other_items[item] += [0 for i in range(old_len)]
            self.other_items[item].append(ratings[idx])

    def compute_sim(self):
        items = ''
        sims = ''
        count = 0
        for item, ratings in self.other_items.items():
            if item == self.item_idx:
                continue
            sim = cos_similarity(self.ratings, ratings)
            if sim >  0 :
                coma = ',' if count > 0 else ''
                items += coma + item
                sims += coma + str(round(sim, 3))
                count += 1
        if items != '':
            print('{}\t{}!{}'.format(self.item_idx, items, sims))

i_prev = None
for line in sys.stdin:
    line = line.strip()
    i, val = line.split('\t')
    if i_prev == None:
        sfi = SimForItem(i)
        i_prev = i
    if i != i_prev:
        i_prev = i
        sfi.compute_sim()
        sfi = SimForItem(i)
    r, items, ratings = val.split('!')
    item_lst = items.split(',')
    rating_lst = np.array(list(map(float, ratings.split(','))))
    sfi.update(float(r), item_lst, rating_lst)

sfi.compute_sim()
