'''This example demonstrates the use of Convolution1D for text classification.

Run on GPU: THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn.py

Get to 0.835 test accuracy after 2 epochs. 100s/epoch on K520 GPU.
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
#from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
#from keras.datasets import imdb


# set parameters:
max_features = 50000
maxlen = 100
batch_size = 32
embedding_dims = 100
#nb_filter = 250
#filter_length = 3
hidden_dims = 250
nb_epoch = 10

print('Loading data...')
data_file1 = "x_0303_training_new.txt"
data_file2 = "x_0303_testing_new.txt"
data_file3 = "y_0303_training.txt"
data_file4 = "y_0303_testing.txt"

# data loading
X_train = np.loadtxt(data_file1, delimiter=',')
print(X_train.shape)
y_train = np.loadtxt(data_file3, delimiter=',')
print(y_train.shape)
X_test = np.loadtxt(data_file2, delimiter=',')
print(X_test.shape)
y_test = np.loadtxt(data_file4, delimiter=',')
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
model.add(Convolution1D(nb_filter=32,
                        filter_length=5,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# we use standard max pooling (halving the output of the previous layer):
#model.add(MaxPooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=2))
model.add(Convolution1D(nb_filter=64,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
# We flatten the output of the conv layer,
# so that we can add a vanilla dense layer:
model.add(Flatten())

# We add a vanilla hidden layer:
#model.add(Dense(hidden_dims))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))

#two FC layers
model.add(Dense(256))
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
