'''Train a recurrent convolutional network on the IMDB sentiment
classification task.

GPU command:
    THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python imdb_cnn_lstm.py

Get to 0.8498 test accuracy after 2 epochs. 41s/epoch on K520 GPU.
'''

from __future__ import print_function
import numpy as np
import pandas as pd
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.datasets import imdb


# Embedding
max_features = 50000
maxlen = 100
embedding_size = 128

# Convolution
filter_length = 3
nb_filter = 64
pool_length = 2

# LSTM
lstm_output_size = 70

# Training
batch_size = 32
nb_epoch = 50

'''
Note:
batch_size is highly sensitive.
Only 2 epochs are needed as the dataset is very small.
'''
print('Loading data...')
data_file1 = "x-3d4hr_0210_training_nor.csv"
data_file2 = "x-3d4hr_0210_testing_nor.csv"
data_file3 = "y-3d4hr_0210_training.csv"
data_file4 = "y-3d4hr_0210_testing.csv"

X_train = pd.read_csv(data_file1, delimiter=',', error_bad_lines=False, header=None)
X_train = X_train.as_matrix()

y_train = pd.read_csv(data_file3, delimiter=',', error_bad_lines=False, header=None)
y_train = y_train.as_matrix()

X_test = pd.read_csv(data_file2, delimiter=',', error_bad_lines=False, header=None)
X_test = X_test.as_matrix()

y_test = pd.read_csv(data_file4, delimiter=',', error_bad_lines=False, header=None)
y_test = y_test.as_matrix()

#(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=max_features, test_split=0.2)

#print(raw_input('123...'))

print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
#X_train = sequence.pad_sequences(X_train, maxlen=maxlen)
#X_test = sequence.pad_sequences(X_test, maxlen=maxlen)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print(X_train)
print(y_train)
#print(raw_input('123...'))


print('Build model...')

model = Sequential()
model.add(Embedding(max_features, embedding_size, input_length=maxlen))
model.add(Dropout(0.25))
model.add(Convolution1D(nb_filter=nb_filter,
                        filter_length=filter_length,
                        border_mode='valid',
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=pool_length))
model.add(LSTM(lstm_output_size))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              class_mode='binary')

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch,
          validation_data=(X_test, y_test), show_accuracy=True)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size,
                            show_accuracy=True)
print('Test score:', score)
print('Test accuracy:', acc)
