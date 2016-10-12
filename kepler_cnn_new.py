from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D, AveragePooling1D, ZeroPadding1D
from keras.utils import np_utils
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score

#THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python kepler_cnn_new.py


# set parameters:
batch_size = 32
input_length= 36
nb_epoch = 50

print('Loading data...')
file1 = "/home/qinghai/research/kepler/1005_1/matrix_f_1006.txt"
data1 = pd.read_csv(file1, sep=',', header=None)
print(data1.shape)

# data loading
X_train = data1.values[:60000, :36, newaxis]
print(X_train.shape)
print(X_train)
y_train = data1.values[:60000, 36]
y_train = y_train.astype('int')
y_train = np_utils.to_categorical(y_train, 2)
print (y_train.shape)

X_test = data1.values[60000:, :36, newaxis]
print(X_test.shape)

y_test = data1.values[60000:, 36]
y_test = y_test.astype('int')
y_test = np_utils.to_categorical(y_test, 2)
print (y_train.shape)
print('Build model...')
model = Sequential()

# I add 10 Convolution1D, which will learn nb_filter
# word group filters of size filter_length:
#model.add(ZeroPadding1D(padding=1))
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
                        
model.add(AveragePooling1D(pool_length=2))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu'))
#flatten the output of the conv layer
model.add(Flatten())

#two FC layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(Dense(128))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(2)) 
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_split=0.2)
score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

classes = model.predict_classes(X_test, batch_size=batch_size)
y_prob = model.predict_proba(X_test, batch_size=batch_size)
print (classes)
print (y_prob)
print (y_prob.shape)
np.savetxt('/home/qinghai/research/kepler/1005_1/prediction_1006.txt', classes, delimiter=',')

fpr, tpr, thresholds = roc_curve(data1.values[60000:, 36], y_prob[:, -1])
roc_auc = auc(fpr, tpr)
print (fpr)
print (tpr)
print (thresholds)
print (roc_auc_score(data1.values[60000:, 36], y_prob[:, -1]))
print (roc_auc)
plt.plot(fpr, tpr)
plt.title('ROC Curve', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.show()
'''
json_string = model.to_json()
open('kepler_cnn_architecture.json', 'w').write(json_string)
model.save_weights('kepler_cnn_weights.h5')
'''
