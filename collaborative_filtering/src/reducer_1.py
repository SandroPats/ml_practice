#!/usr/bin/python3

import sys

u_prev = None
items = []
ratings = []
for line in sys.stdin:
    line = line.strip()
    u, value = line.split('\t')
    i, r = value.split('!')
    if u_prev is None:
        u_prev = u

    if u != u_prev:
        print('{}\t{}!{}'.format(u_prev, ','.join(items), ','.join(ratings)))
        u_prev = u
        items = []
        ratings = []

    ratings.append(r)
    items.append(i)

print('{}\t{}!{}'.format(u_prev, ','.join(items), ','.join(ratings)))
