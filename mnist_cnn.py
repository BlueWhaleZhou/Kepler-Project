#Implementation of CNN with MNIST dataset for Kaggle Competition
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, AveragePooling2D
from keras.utils import np_utils
from keras import backend as K
import pandas as pd

#model parameters
batch_size = 64
nb_classes = 10
nb_epoch = 20
nb_filters = 512
#data dimension
img_rows = 28
img_cols = 28
kernel_size = (3, 3)
pooling_size = (2, 2)

#loading mnist data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
X_train = train.values[:, 1:]
Y_train = train.values[:, 0]
X_test = test.values[:, :]

#reshaping
X_train = X_train.reshape(X_train.shape[0], 1, img_rows, img_cols)
X_test = X_test.reshape(X_test.shape[0], 1, img_rows, img_cols)
input_shape = (1, img_rows, img_cols)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= X_train
X_test /= X_test

Y_train = np_utils.to_categorical(Y_train, nb_classes)

#model
model = Sequential()
model.add(Convolution2D(nb_filters=nb_filters, kernel_size[0], kernel_size[1], border_mode='valid', input_shape = input_shape))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters=nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=pooling_size))
model.add(Dropout)
model.add(Convolution2D(nb_filters=nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))
model.add(Convolution2D(nb_filters=nb_filters, kernel_size[0], kernel_size[1]))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
model.add(Dense(nb_classes))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adadelta',metrics=['accuracy'])
model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1)

Y_prediction = model.predict_classes(X_test, batch_size=batch_size, verbose=1)
pd.Dataframe({"ImageId": range(1, len(Y_test) + 1), "Label": Y_prediction}).to_csv('out.csv', index=False, header=True)
