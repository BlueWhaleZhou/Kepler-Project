from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D



# set parameters:
batch_size = 16
input_length=15
nb_epoch = 10

print('Loading data...')
data_file1 = "/home/qinghai/kepler/Kepler/x_0329.txt"
data_file2 = "/home/qinghai/kepler/Kepler/y-0321.csv"

# data loading
X = np.loadtxt(data_file1, delimiter=',')
X = X[:, :, newaxis]
X_train = X[:200000, :, :]
X_test = X[200000:240000, :, :]
print(X_train.shape)
print(X_test.shape)

y = pd.read_csv(data_file2, delimiter=',', error_bad_lines=False, header=None)
y = y.as_matrix()
y_train = y[:200000, :]
y_test = y[200000:240000, :]

print(y_train.shape)
print(y_test.shape)
'''
X_test = np.loadtxt(data_file2, delimiter=',')
X_test = X_test[:, :, newaxis]
print(X_test.shape)

y_test = pd.read_csv(data_file4, delimiter=',', error_bad_lines=False, header=None)
y_test = y_test.as_matrix()
print(y_test.shape)
'''
print('Build model...')
model = Sequential()

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
			input_dim=1, 
			input_length=input_length)) 
#model.add(ZeroPadding1D(padding=1))                       
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
                        
#model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
                    
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
'''
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
'''
#model.add(MaxPooling1D(pool_length=2))
# We flatten the output of the conv layer,
model.add(Flatten())

#two FC layers
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(128))
model.add(Dropout(0.25))
model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))

