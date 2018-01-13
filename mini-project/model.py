'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

import gpflow
from sklearn.metrics import accuracy_score

import numpy as np

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols = 28, 28

# the data, shuffled and split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()

if K.image_data_format() == 'channels_first':
    x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
    x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
    input_shape = (1, img_rows, img_cols)
else:
    x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
    x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
    input_shape = (img_rows, img_cols, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape, 'test samples')

# convert class vectors to binary class matrices
y_train_m = keras.utils.to_categorical(y_train, num_classes)
y_test_m = keras.utils.to_categorical(y_test, num_classes)
print(y_test_m.shape)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train_m,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test_m))

features_extractor = Model(inputs=model.input, outputs=model.layers[-2].output)
features_train = features_extractor.predict(x_train).astype(np.float64)


kernel = gpflow.kernels.Matern32(1) + gpflow.kernels.White(1, variance=0.01)
"""gaussian_classifier = gpflow.models.SVGP(features_train, y_train,
                                         kernel,
                                         likelihood=gpflow.likelihoods.MultiClass(10),
                                         Z=features_train[::10].copy(), num_latent=10, whiten=True, q_diag=True)"""
print("FITTED")

features_test = features_extractor.predict(x_test).astype(np.float64)
#y_predict = gaussian_classifier.predict(features_test)

score_traditional = model.evaluate(x_test, y_test_m, verbose=0)
#score_gaussian = accuracy_score(y_test[:1000], y_predict)

print('Standard CNN accuracy:', score_traditional)
#print('CNN + Gaussian Process accuracy', score_gaussian)
