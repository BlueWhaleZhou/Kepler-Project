from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
from keras.layers.recurrent import LSTM, GRU, SimpleRNN
from keras.utils import np_utils
from sklearn.metrics import roc_curve, auc, roc_auc_score

# set parameters:
batch_size = 16
input_dim = 1
nb_epoch = 50
input_length = 36

print('Loading data...')
file1 = "/home/qinghai/research/kepler/1005_1/matrix_f_1006.txt"

# data loading
data1 = pd.read_csv(file1, sep=',', header=None)
X_train = data1.values[:60000, :36, newaxis]
print(X_train.shape)

y_train = data1.values[:60000, 36]
y_train = y_train.astype('int')
y_train = np_utils.to_categorical(y_train, 2)
print(y_train.shape)

X_test = data1.values[60000:, :36, newaxis]
print(X_test.shape)

y_test = data1.values[60000:, 36]
y_test = y_test.astype('int')
y_test = np_utils.to_categorical(y_test, 2)
print(y_test.shape)

print('Build model...')
model = Sequential()

#rnn layer
model.add(GRU(output_dim=64, input_dim=input_dim))
model.add(Dropout(0.25))
model.add(Activation('relu'))
#two FC layers
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))

model.add(Dense(2)) 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta', metrics=['accuracy'])

print('Train...')
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_data=(X_test, y_test))
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)
classes = model.predict_classes(X_test, batch_size=batch_size)
print (classes)
y_prob = model.predict_proba(X_test, batch_size=batch_size)
print (y_prob)
print (y_prob.shape)
np.savetxt('/home/qinghai/research/kepler/1005_1/predition_1006_rnn.txt', classes, delimiter=',')

fpr, tpr, thresholds = roc_curve(y_test, y_prob[:, -1])
roc_auc = auc(fpr, tpr)
print (fpr)
print (tpr)
print (thresholds)
print (roc_auc_score(y_test, y_prob[:, -1]))
print (roc_auc)
plt.plot(fpr, tpr)
plt.title('ROC Curve', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.show()

