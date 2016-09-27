import numpy as np
import os
import glob
import csv

#generating starlist in order

name = []
star_list = []
path_flux = '/home/qinghai/testing/flux/'
for filename in glob.glob(os.path.join(path_flux, '*.txt')):
    dir_len = len(path_flux)
    name_tmp = filename[dir_len:].split('_')
    name.append(name_tmp[0])

print name
for i in range(len(name)):
    filepath_flux_temp = '/home/qinghai/testing/flux/' + name[i] + '_flux.txt'
    flux_matrix_tmp = np.loadtxt(filepath_flux_temp, delimiter=',')
    length = len(flux_matrix_tmp)
    for j in range(length):
        star_list.append(name[i])
print star_list

with open("star_list.csv", "wb") as f:
    writer = csv.writer(f, delimiter=',')
    writer.writerows([item.split(',') for item in star_list])
