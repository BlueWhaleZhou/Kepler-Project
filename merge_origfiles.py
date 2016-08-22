import numpy as np
import os
import glob
from sklearn import preprocessing
import pandas as pd

path_flux = "/home/qinghai/testing/flux/"

name = []

for filename in glob.glob(os.path.join(path_flux, '*.txt')):
    dir_len = len(path_time)
    name_tmp = filename[dir_len:].split('_')
    name.append(name_tmp[0])

print name
#concatenate all time and flux files
filepath_time_temp = '/home/qinghai/testing/time/' + name[0] + '_time.txt'
time_matrix = np.loadtxt(filepath_time_temp, delimiter = ',')
filepath_flux_temp = '/home/qinghai/testing/flux/' + name[0] + '_flux.txt'
flux_matrix = np.loadtxt(filepath_temp, delimiter = ',')
i = 1
while i < len(name):
    filepath_time_temp = '/home/qinghai/testing/time/' + name[i] + '_time.txt'
    filepath_flux_temp = '/home/qinghai/testing/flux/' + name[i] + '_flux.txt'
    print filepath_time_temp
    print filepath_flux_temp
    time_matrix_tmp = np.loadtxt(filepath_time_temp, delimiter=',')
    flux_matrix_tmp = np.loadtxt(filepath_flux_temp, delimiter=',')
    time_matrix = np.concatenate((time_matrix, time_matrix_tmp), axis = 0)
    flux_matrix = np.concatenate((flux_matrix, flux_matrix_tmp), axis = 0)
    i = i + 1
print time_matrix.shape
print flux_matrix.shape



