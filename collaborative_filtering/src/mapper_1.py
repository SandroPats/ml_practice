#!/usr/bin/python3

import sys

for i, line in enumerate(sys.stdin):
    if i == 0:
        continue
    line = line.strip()
    u, i, r, t = line.split(',')
    r = float(r) / 5
    print('{}\t{}!{:.1f}'.format(u, i, r))
