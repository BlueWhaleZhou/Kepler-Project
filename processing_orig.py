#processing kepler original data
from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import math

def processingoriginal():
    path1='/home/qinghai/kepler/kepler_0021_orig/'

    step = 20
    name1 = []
    star = dict()

    for filename in glob.glob(os.path.join(path1,'*.fits')):
        name1.append(filename)
 #   print name1

#sorting in star names
    for filename in glob.glob(os.path.join(path1, '*.fits')):
        dir_len = len(path1)
        name_tmp = filename[dir_len:].split('_')
        star_name = name_tmp[0]
#        print star_name
        num_tmp = int(name_tmp[1][1:3])
#        print num_tmp
        if star_name in star:
            star[star_name] = max(star[star_name], num_tmp)
        else:
            star[star_name] = num_tmp
    print star
#    input('break0...')
#extracting time and flux sequences
    for key in star:
        print key
        time = []
        flux = []
        max_num = star[key]
        for i in xrange(1, max_num+1):
            if i >= 10:
                file_name = key + "_q" + str(i) + '.fits'
            else:
                file_name = key + "_q0" + str(i) + '.fits'
            data2 = fits.getdata('/home/qinghai/kepler/kepler_0021_orig/' + file_name)
            time.extend(data2[0])
            flux.extend(data2[1])
        rows = len(time) / step
#        print rows
        new_length = rows * step
        time = time[:new_length]
        flux = flux[:new_length]
        time_matrix = np.reshape(time, (rows, step))
        flux_matrix = np.reshape(flux, (rows, step))
#        print time_matrix
#        print flux_matrix
        print time_matrix.shape
        print flux_matrix.shape
#        input('break1...')
        np.savetxt(key + '_time.txt', time_matrix, delimiter = ',')
        np.savetxt(key + '_flux.txt', flux_matrix, delimiter = ',')
processingoriginal()
