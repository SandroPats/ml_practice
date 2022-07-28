#!/usr/bin/python3

import sys
import pandas as pd

movies = pd.read_csv('movies.csv', usecols=[0, 1], index_col=[0])['title']

for line in sys.stdin:
    line = line.strip()
    u, r, i = line.split('\t')
    title = movies[int(i)]
    print('{}\t{}\t{}\t'.format(u, r, title))
