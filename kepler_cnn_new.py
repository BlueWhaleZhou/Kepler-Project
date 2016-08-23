from __future__ import print_function
import numpy as np
from numpy import newaxis
import pandas as pd
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD
from keras.layers.convolutional import Convolution1D, MaxPooling1D, ZeroPadding1D
import matplotlib.pyplot as plt
from keras.models import model_from_json

#from sklearn.metrics import roc_curve, auc, roc_auc_score

#THEANO_FLAGS=mode=FAST_RUN,device=gpu2,floatX=float32 python kepler_cnn_new.py


# set parameters:
batch_size = 16
input_length= 50
nb_epoch = 100

print('Loading data...')
file1 = "/home/qinghai/kepler/data/0021_2015Nov12/0021_2015Nov12_0427_One_new.txt"
data1 = np.loadtxt(file1, delimiter=',')
# data loading
X_train = data1[:50000, :50, newaxis]
print(X_train.shape)

y_train = data1[:50000, -1]
print(y_train.shape)

X_test = data1[50000:, :50, newaxis]
print(X_test.shape)

y_test = data1[50000:, -1]
print(y_test.shape)

print('Build model...')
model = Sequential()

model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        border_mode='valid',
                        activation='relu',
						input_dim=1, 
						input_length=input_length)) 
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
#flatten the output of the conv layer
model.add(Flatten())

#two FC layers
model.add(Dense(128))
model.add(Activation('relu'))
model.add(Dropout(0.25))
#model.add(Dense(128))
#model.add(Dropout(0.25))
#model.add(Activation('relu'))

# We project onto a single unit output layer, and squash it with a sigmoid:
model.add(Dense(1)) 
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adadelta',
              metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=batch_size,
          nb_epoch=nb_epoch, validation_data=(X_test, y_test))

score, acc = model.evaluate(X_test, y_test, batch_size=batch_size)
print('Test score:', score)
print('Test accuracy:', acc)

# serialize model to JSON
model_json = model.to_json()
with open("/home/qinghai/testing/kepler_model.json", "w") as json_file:
    json_file.write(model_json)

# serialize weights to HDF5
model.save_weights("/home/qinghai/testing/kepler_model.h5")

classes = model.predict_classes(X_test, batch_size=batch_size)
#y_prob = model.predict_proba(X_test, batch_size=batch_size)
#print(y_prob)
print(classes)

'''
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
print (fpr)
print (tpr)
print (thresholds)
print (roc_auc_score(y_test, y_prob))
print (roc_auc)
plt.plot(fpr, tpr)
plt.title('ROC Curve', fontsize=14)
plt.xlabel('FPR', fontsize=14)
plt.ylabel('TPR', fontsize=14)
plt.show()
np.savetxt('superEarth_prediction.txt', classes, delimiter=',')
np.savetxt('superEarth_probability.txt', y_prob, delimiter=',')
json_string = model.to_json()
open('kepler_cnn_architecture.json', 'w').write(json_string)
model.save_weights('kepler_cnn_weights.h5')
'''
