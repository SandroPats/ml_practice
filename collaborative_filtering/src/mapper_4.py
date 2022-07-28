#!/usr/bin/python3

import sys

for line in sys.stdin:
    line = line.strip()
    if '!' in line:
        print(line)
    else:
        u, i, r, t = line.split(',')
        if u == 'userId':
            continue
        print('{}!{}\t_'.format(u, i))
