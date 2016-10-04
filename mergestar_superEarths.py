import numpy as np
import pandas as pd
from sklearn import preprocessing
import glob
from numpy import newaxis
import os

period = 243
path = '/home/qinghai/research/kepler/' + str(period) + 'days/'
for filename in glob.glob(os.path.join(path, '*.txt')):
    starlist = []
    nan_index = []
    dir_len = len(path)
    name_temp = filename[dir_len:].split('.')
    starname = name_temp[0]
    print starname
    matrix = np.loadtxt(filename, delimiter=',')
    for i in range(len(matrix)):
        starlist.append(starname)
        if np.isnan(matrix[i][3]) and np.isnan(matrix[i][4]) and np.isnan(matrix[i][5]) and np.isnan(matrix[i][19]) and np.isnan(matrix[i][20]):
            nan_index.append(i)
    print len(nan_index)
    print len(matrix)
    matrix = np.delete(matrix, nan_index, axis=0)
    print len(matrix)
    starlist = starlist[:len(matrix)]
    data = pd.DataFrame(matrix[:, :28])
    data_mean = data.mean(axix=0)
    data = data.fillna(data_mean, axis=0)
    data_normalized = preprocessing.normalize(data, norm='l1')
    data_normalized = 10000 * (data_normalized - 0.0357)
    starlist = np.asarray(starlist)
    starlist = starlist[:, newaxis]
    print starlist.shape
    matrix_after = np.concatenate((np.concatenate((data_normalized, matrix[:, 28:]), axis=1), starlist), axis=1)
    print matrix_after.shape
    matrix_after = pd.DataFrame(matrix_after)
    name = str(starname) + '_norm.txt'
    print name
    matrix_after.to_csv(name, sep=',', columns=None, index=False)

