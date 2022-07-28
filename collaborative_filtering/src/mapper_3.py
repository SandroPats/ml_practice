#!/usr/bin/python3

import sys

for line in sys.stdin:
    line = line.strip()
    if '\t' in line:
        i, val = line.split('\t')
        items, sims = val.split('!')
        items = items.split(',')
        sims = sims.split(',')
        for item, sim in zip(items, sims):
            print('{}\t{}!{}'.format(i, item, sim))
    else:
        if 'userId' in line:
            continue
        u, i, r, t = line.split(',')
        print('{}\t{},{:.1f}'.format(i, u, float(r)/5))
