import numpy as np
from random import randint
from numpy import newaxis
import random
# Generating periodical data(30000x400) for training

a = np.random.randint(low=50, high=150, size=50)
print(a)
print(len(a))
sequence_temp = np.zeros(400000)
k = int(random.random()*a[0])
while(k < 400000):
	sequence_temp[k] = sequence_temp[k] + 1
	k = k + a[0]
sequence_temp = np.reshape(sequence_temp, (800, 500))
print(sequence_temp.shape)
#print(sequence_temp)
j = 1
for j in range(1, 50):
	i = int(random.random()*a[j])
	sequence_new = np.zeros(400000)
	while(i < 400000):
        	sequence_new[i] = sequence_new[i] + 1
        	i = i + a[j]
	sequence_new = np.reshape(sequence_new, (800, 500))
	sequence_temp = np.concatenate((sequence_temp, sequence_new), axis=0)
#print(sequence_temp)
print(sequence_temp.shape)
label_p = np.zeros(40000)
label_p = label_p + 1
label_p = label_p[:, newaxis]
training_p = np.concatenate((sequence_temp, label_p), axis=1)
print(training_p.shape)
#print(training_p)
#x = input('...')
np.savetxt('training_p.txt', training_p, delimiter=',', header='')
#x = input('...')

#Generating while noise(15000x400)

noise = np.zeros(2500000)
i = randint(1, 20)
for i in range(0, 2500000):
        noise[i] = randint(0, 1)
#print(noise)
print(noise.shape)
noise = np.reshape(noise, (5000, 500))
print(noise.shape)
label_n= np.zeros(5000)
label_n = label_n[:, newaxis]
training_n = np.concatenate((noise, label_n), axis=1)
print(training_n.shape)
#x = input('...')
np.savetxt('training_n.txt', training_n, delimiter=',')

#Generating slightly different 'non-periodical' data(15000x400)
b = np.random.randint(low=50, high=150, size=50)
print(b)
print(len(b))
sequence_temp = np.zeros(250000)
k = int(random.random()*b[0])
while(k < 250000):
        sequence_temp[k] = sequence_temp[k] + 1
        k = k + b[0] + randint(0, 3)
sequence_temp = np.reshape(sequence_temp, (500, 500))
print(sequence_temp.shape)
#print(sequence_temp)
l = 1
for l in range(1, 50):
        m = int(random.random()*b[l])
        sequence_new = np.zeros(250000)
        while(i < 250000):
                sequence_new[m] = sequence_new[m] + 1
                m = m + b[l] + randint(0, 3)
        sequence_new = np.reshape(sequence_new, (500, 500))
        sequence_temp = np.concatenate((sequence_temp, sequence_new), axis=0)
#print(sequence_temp)
print(sequence_temp.shape)
label_s = np.zeros(25000)
label_s = label_s[:, newaxis]
training_s = np.concatenate((sequence_temp, label_s), axis=1)
print(training_s.shape)
#print(training_s)
np.savetxt('training_s.txt', training_s, delimiter=',', header='')

temp_matrix = np.concatenate((training_p, training_n), axis=0)
matrix = np.concatenate((temp_matrix, training_s), axis=0)
matrix = np.random.permutation(matrix)
print(matrix.shape)
np.savetxt('training.txt', matrix, delimiter=',')

#Generating periodical data with 1 label

c = np.random.randint(low=50, high=250, size=50)
print(len(a))
sequence_temp = np.zeros(100000)
k = int(random.random()*c[0])
while(k < 100000):
        sequence_temp[k] = sequence_temp[k] + 1
        k = k + c[0]
sequence_temp = np.reshape(sequence_temp, (200, 500))
print(sequence_temp.shape)
#print(sequence_temp)
j = 1
for j in range(1, 50):
        i = int(random.random()*c[j])
        sequence_new = np.zeros(100000)
        while(i < 100000):
                sequence_new[i] = sequence_new[i] + 1
                i = i + c[j]
        sequence_new = np.reshape(sequence_new, (200, 500))
        sequence_temp = np.concatenate((sequence_temp, sequence_new), axis=0)
print(sequence_temp)
print(sequence_temp.shape)
label_p0 = np.zeros(10000)
label_p0 = label_p0 + 1
label_p0 = label_p0[:, newaxis]
training_p0 = np.concatenate((sequence_temp, label_p0), axis=1)
print(training_p0.shape)
#print(training_p0)
#x = input('...')
np.savetxt('training_p0.txt', training_p0, delimiter=',', header='')
#x = input('...')
testing = np.concatenate((matrix[50000:70000, :], training_p0), axis=0)
testing = np.random.permutation(testing)
print(testing.shape)
print(testing)
np.savetxt('testing.txt', testing, delimiter=',')
