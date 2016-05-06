from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import glob
import math

def loaddata():
	path1='/home/qinghai/0021_2015Nov12/giant/'

	step = 50
	name1 = []
	period = 3
	star = dict()

#length estimation
	for filename in glob.glob(os.path.join(path1,'*.fits')):
		name1.append(filename)
	summ = 0
	for i in range(len(name1)):
		data1 = fits.getdata(name1[i])
		time = data1[0]
		length = len(time)
		summ = summ + int(math.floor(len(time)/step)-1)
	print (summ)
#	x = input('...')
#loading data into matrix
	for filename in glob.glob(os.path.join(path1, '005780885*.fits')):
		dir_len = len(path1)
		name_tmp = filename[dir_len:].split('_')
		star_name = name_tmp[0]
		num_tmp = int(name_tmp[1][1:3])
		if star_name in star:
			star[star_name] = max(star[star_name], num_tmp)
		else:
			star[star_name] = num_tmp
	s = (step, 120589)
	trainx = np.zeros(s)
	trainy = np.zeros(120589)
	il = 0
	index = 0
	for key in star:
		print key
		time = []
		flux = []
		max_num = star[key]
		for i in xrange(1,max_num+1):
			if i>=10:		
				file_name = key+"_q"+str(i)+'.fits'
			else:
				file_name = key+"_q0"+str(i)+'.fits'
			data2 = fits.getdata('/home/qinghai/0021_2015Nov12/giant/'+file_name)
			time.extend(data2[0])
			flux.extend(data2[1])
		initialtime = time[0] + period
		k = 0
		t = 0
		transits = []
		while(k < time[-1]):
			k = initialtime+t*period
			transits.append(k)
			t = t + 1
		initial = 0
		for j in range(int(math.floor(len(time)/step)-1)):
			fluxtem = flux[initial:initial+step]
			timetem = time[initial:initial+step]
			plt.plot(timetem, fluxtem)
			plt.show()
			trainy[j+index] = 0
			for m in range(len(transits)):
				if(time[initial] < transits[m] and transits[m] < time[initial + step]):
					trainy[j+index] = 1
#print trainy[j+index]
			trainx[:, j+index] = fluxtem
			initial = initial + step
		index = index + int(math.floor(len(time)/step)-1)
	print j + index
#	x = input('...')
	p = 'x_0021Nov12_243days.txt'
	q = 'y_0021Nov12_243days.txt'
	np.savetxt(p, trainx, delimiter=',')
loaddata()
