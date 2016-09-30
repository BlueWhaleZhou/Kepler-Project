#processing kepler original data
from astropy.io import fits
import numpy as np
from numpy import newaxis
import matplotlib.pyplot as plt
import os
import glob
import math

def processingsuperEarth():
    step = 28
    star = dict()
    period = 3
    path='/home/qinghai/research/kepler/' + str(period) + 'days/'

#sorting in star names
    for filename in glob.glob(os.path.join(path, '*.fits')):
        dir_len = len(path)
        name_tmp = filename[dir_len:].split('_')
        star_name = name_tmp[0]
        num_tmp = int(name_tmp[1][1:3])
        if star_name in star:
            star[star_name] = max(star[star_name], num_tmp)
        else:
            star[star_name] = num_tmp
    print star
#extracting time and flux sequences
    for key in star:
        print key
        time = []
        flux = []
        position = []
        max_num = star[key]
        for i in xrange(1, max_num+1):
            if i >= 10:
                file_name = key + "_q" + str(i) + '.fits'
            else:
                file_name = key + "_q0" + str(i) + '.fits'
            data = fits.getdata(path + file_name)
            time.extend(data[0])
            flux.extend(data[1])
        rows = len(time) / step
        new_length = rows * step
        time = time[:new_length]
        time_tmp = time[0] + period
        position.append(time_tmp)
        while (time_tmp < time[-1]):
            time_tmp += period
            position.append(time_tmp)
        print position
        print len(position)
        flux = flux[:new_length]
        time_matrix = np.reshape(time, (rows, step))
        flux_matrix = np.reshape(flux, (rows, step))
        labels = np.zeros(rows)
        for i in range((len(time_matrix))):
            for j in range(len(position)):
                if(time_matrix[i][0] <= position[j] and position[j] <= time_matrix[i][-1])
                    labels[i] = 1
                    break
        labels = labels[:, newaxis]
        print labels.shape
        final_matrix = np.concatenate(np.concatenate(flux_matrix, labels, axis=1), time_matrix, axis=1)
        np.savetxt(key + '.txt', final_matrix, delimiter = ',')
processingsuperEarth()
