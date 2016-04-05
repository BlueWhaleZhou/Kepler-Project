import numpy as np
from random import randint
from numpy import newaxis

# Generating periodical data(30000x400) for training

a = ([3, 7, 8, 10, 12, 14, 15, 18, 21, 24, 27, 33, 36, 39, 42, 45, 47, 50, 52, 55, 59, 61, 67, 68, 71, 72, 75, 76, 78, 80, 82, 85, 87, 90, 92, 95, 98, 66, 29, 31])
print(len(a))
sequence_temp = np.zeros(400000)
k = randint(1, 20)
while(k < 400000):
	sequence_temp[k] = sequence_temp[k] + 1
	k = k + a[0]
sequence_temp = np.reshape(sequence_temp, (1000, 400))
print(sequence_temp.shape)
#print(sequence_temp)
j = 1
for j in range(1, 40):
	i = randint(0, 20)
	sequence_new = np.zeros(400000)
	while(i < 400000):
        	sequence_new[i] = sequence_new[i] + 1
        	i = i + a[j]
	sequence_new = np.reshape(sequence_new, (1000, 400))
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

noise = np.zeros(6000000)
i = randint(1, 20)
for i in range(0, 6000000):
        noise[i] = randint(0, 1)
#print(noise)
print(noise.shape)
noise = np.reshape(noise, (15000, 400))
print(noise.shape)
label_n= np.zeros(15000)
label_n = label_n[:, newaxis]
training_n = np.concatenate((noise, label_n), axis=1)
print(training_n.shape)
#x = input('...')
np.savetxt('training_n.txt', training_n, delimiter=',')

#Generating slightly different 'non-periodical' data(15000x400)
b = ([3, 7, 10, 14, 18, 21, 24, 27, 33, 39, 42, 45, 50, 52, 55, 59, 61, 67, 71, 75, 78, 80, 82, 87, 90, 92, 95, 66, 29, 31])
print(len(b))
sequence_temp = np.zeros(200000)
k = randint(1, 20)
while(k < 200000):
        sequence_temp[k] = sequence_temp[k] + 1
        k = k + b[0] + randint(0, 3)
sequence_temp = np.reshape(sequence_temp, (500, 400))
print(sequence_temp.shape)
#print(sequence_temp)
l = 1
for l in range(1, 30):
        m = randint(0, 20)
        sequence_new = np.zeros(200000)
        while(i < 200000):
                sequence_new[m] = sequence_new[m] + 1
                m = m + b[l] + randint(0, 3)
        sequence_new = np.reshape(sequence_new, (500, 400))
        sequence_temp = np.concatenate((sequence_temp, sequence_new), axis=0)
#print(sequence_temp)
print(sequence_temp.shape)
label_s = np.zeros(15000)
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

#Generating periodical data with 0 label

c = ([5, 9, 11, 17, 23, 25, 28, 30, 35, 38, 40, 44, 48, 54, 57, 62, 65, 73, 88, 96])
print(len(a))
sequence_temp = np.zeros(160000)
k = randint(1, 20)
while(k < 160000):
        sequence_temp[k] = sequence_temp[k] + 1
        k = k + c[0]
sequence_temp = np.reshape(sequence_temp, (400, 400))
print(sequence_temp.shape)
#print(sequence_temp)
j = 1
for j in range(1, 20):
        i = randint(0, 20)
        sequence_new = np.zeros(160000)
        while(i < 160000):
                sequence_new[i] = sequence_new[i] + 1
                i = i + c[j]
        sequence_new = np.reshape(sequence_new, (400, 400))
        sequence_temp = np.concatenate((sequence_temp, sequence_new), axis=0)
print(sequence_temp)
print(sequence_temp.shape)
label_p0 = np.zeros(8000)
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
np.savetxt('testing.txt', testing, delimiter=',')
