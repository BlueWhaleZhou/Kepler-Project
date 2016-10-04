import numpy as np
import os
import glob
import pandas as pd

starlist = []
path = '/home/qinghai/research/kepler/1001/'
for filename in glob.glob(os.path.join(path, '*_norm.txt')):
    dir_len = len(path)
    nametemp = filename[dir_len:].split('_')
    starname = nametemp[0]
    starlist.append(starname)
print starlist
print len(starlist)
matrix = pd.read_csv(str(starlist[0]) + '_norm.txt', sep=',')
for i in range(1, len(starlist)):
    matrix_temp = pd.read_csv(str(starlist[i]) + '_norm.txt', sep=',')
    matrix = np.concatenate((matrix, matrix_temp), axis=0)
print matrix.shape

matrix = np.random.permutation(matrix)
print matrix.shape
matrix = pd.DataFrame(matrix)
matrix.to_csv('matrix_total.txt', sep=',', columns=None, index=False)
