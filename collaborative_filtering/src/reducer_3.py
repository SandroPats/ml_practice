#!/usr/bin/python3

import sys

def red(u_r_lst, i_s_lst):
    for u, r in u_r_lst:
        for item, sim in i_s_lst:
            print('{}!{}\t{}!{}'.format(u, item, sim, r))

u_r_lst = []
i_s_lst = []
prev_i = None
for line in sys.stdin:
    i, val = line.split()
    if prev_i == None:
        prev_i = i
    if i != prev_i:
        prev_i = i
        red(u_r_lst, i_s_lst)
        u_r_lst = []
        i_s_lst = []

    if '!' in val:
        item, sim = val.split('!')
        i_s_lst.append((item, sim))
    else:
        u, r = val.split(',')
        u_r_lst.append((u, r))

red(u_r_lst, i_s_lst)
