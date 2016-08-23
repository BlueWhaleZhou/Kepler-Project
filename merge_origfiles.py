import numpy as np
import os
import glob
from sklearn import preprocessing
import pandas as pd

path_flux = "/home/qinghai/testing/flux/"

name = []

for filename in glob.glob(os.path.join(path_flux, '*.txt')):
    dir_len = len(path_flux)
    name_tmp = filename[dir_len:].split('_')
    name.append(name_tmp[0])

print name

#concatenate all time and flux files
filepath_time_temp = '/home/qinghai/testing/time/' + name[0] + '_time.txt'
time_matrix = np.loadtxt(filepath_time_temp, delimiter=',')
filepath_flux_temp = '/home/qinghai/testing/flux/' + name[0] + '_flux.txt'
flux_matrix = np.loadtxt(filepath_time_temp, delimiter=',')
i = 1
while i < len(name):
    filepath_time_temp = '/home/qinghai/testing/time/' + name[i] + '_time.txt'
    filepath_flux_temp = '/home/qinghai/testing/flux/' + name[i] + '_flux.txt'
    print filepath_time_temp
    print filepath_flux_temp
    time_matrix_tmp = np.loadtxt(filepath_time_temp, delimiter=',')
    flux_matrix_tmp = np.loadtxt(filepath_flux_temp, delimiter=',')
    time_matrix = np.concatenate((time_matrix, time_matrix_tmp), axis=0)
    flux_matrix = np.concatenate((flux_matrix, flux_matrix_tmp), axis=0)
    i = i + 1
#print time_matrix.shape
#print flux_matrix.shape

np.savetxt('kepler_orig_testing_i.txt', flux_matrix, delimiter=',')
flux_file = 'kepler_orig_testing_i.txt'
flux_dataframe = pd.read_csv(flux_file, delimiter=',', header=None, error_bad_lines=False)
flux_dataframe_mean = flux_dataframe.mean(axis=1)
flux_dataframe = flux_dataframe.fillna(flux_dataframe_mean, axis=1)
flux_dataframe = flux_dataframe.fillna(0, axis=1)
flux_matrix_norm = preprocessing.normalize(flux_dataframe, norm=l2, axis=1, copy=True)
flux_matrix_f = (flux_matrix_norm - 0.2236) * 1000000
flux_matrix_f = flux_matrix_f.as_matrix()
print flux_matrix_f.shape
print flux_matrix_f

np.savetxt('kepler_orig_testing_f.txt', flux_matrix_f, delimiter=',')