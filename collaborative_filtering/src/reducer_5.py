#!/usr/bin/python3

import sys

top_lst = []
prev_u = None
for line in sys.stdin:
    line = line.strip()
    u, r, title = line.split('\t')
    if prev_u == None:
        prev_u = u
    if prev_u != u:
        print('{}@{}'.format(prev_u, '@'.join(top_lst)))
        prev_u = u
        top_lst = []
    elif len(top_lst) == 100:
        continue
    pair = '{:.3f}%{}'.format(float(r), title)
    top_lst.append(pair)

print('{}@{}'.format(u, '@'.join(top_lst)))
