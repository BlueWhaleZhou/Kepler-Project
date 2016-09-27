import numpy as np
import os
import glob
import csv

#generating starlist in order

name = []

path_flux = '/home/qinghai/testing/flux/'
for filename in glob.glob(os.path.join(path_flux, '*.txt')):
    dir_len = len(path_flux)
    name_tmp = filename[dir_len:].split('_')
    name.append(name_tmp[0])

print name

filepath_flux_temp = '/home/qinghai/testing/flux/' + name[0] + '_flux.txt'
flux_matrix = np.loadtxt(filepath_flux_temp, delimiter=',')


