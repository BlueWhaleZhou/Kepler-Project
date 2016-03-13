from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
#from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.datasets import imdb


# set parameters:
#max_features = 50000
#maxlen = 100
batch_size = 32
#embedding_dims = 100
#nb_filter = 250
#filter_length = 3
#hidden_dims = 250
input_length=100
nb_epoch = 10

print('Loading data...')
data_file1 = "x_0303_training_new.txt"
data_file2 = "x_0303_testing_new.txt"
data_file3 = "y_0303_training.csv"
data_file4 = "y_0303_testing.csv"

# data loading
X_train = np.loadtxt(data_file1, delimiter=',')
X_train = X_train[:, :, newaxis]
print(X_train.shape)

y_train = pd.read_csv(data_file3, delimiter=',', error_bad_lines=False, header=None)
y_train = y_train.as_matrix()
print(y_train.shape)

X_test = np.loadtxt(data_file2, delimiter=',')
X_test = X_test[:, :, newaxis]
print(X_test.shape)

y_test = pd.read_csv(data_file4, delimiter=',', error_bad_lines=False, header=None)
y_test = y_test.as_matrix()
print(y_test.shape)

#print(y_train)
#print(y_test)

#char = input("...")	
	
#print('Pad sequences (samples x time)')
#X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
#print('X_train shape:', X_train.shape)
#print('X_test shape:', X_test.shape)

print('Build model...')
model = Sequential()

# we start off with an efficient embedding layer which maps
# our vocab indices into embedding_dims dimensions
#model.add(Embedding(max_features, embedding_dims, input_length=maxlen))
#model.add(Dropout(0.25))

# we add a Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
model.add(Convolution1D(nb_filter=64,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
			input_dim=1, 
			input_length=input_length))                        
model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
                        
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=96,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
                        
model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
                        
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

#two FC layers
model.add(Dense(256), activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(256, activation('relu'))
model.add(Dropout(0.25))
# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1), activation('sigmoid'))
#sgd = SGD(lr=0.0001, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              class_mode='binary')
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, show_accuracy=True,
          validation_data=(X_test, y_test))
