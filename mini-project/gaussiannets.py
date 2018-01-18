'''Trains a simple convnet on the MNIST dataset.

Gets to 99.25% test accuracy after 12 epochs
(there is still a lot of margin for parameter tuning).
16 seconds per epoch on a GRID K520 GPU.
'''

import tensorflow as tf
import keras
from keras.datasets import mnist
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D

import gpflow
from sklearn.metrics import accuracy_score

import numpy as np

class ProbabilisticCNN:
    """
    """
    def __init__(self, input_shape, num_classes, batch_size = 128, epochs = 9):
        self.__num_classes = num_classes
        self.__batch_size = batch_size
        self.__epochs = epochs

        self.__network = Sequential()
        self.__network.add(Conv2D(32, kernel_size=(3, 3),
                                      activation='relu',
                                      input_shape=input_shape))
        self.__network.add(Conv2D(64, (3, 3), activation='relu'))
        self.__network.add(MaxPooling2D(pool_size=(2, 2)))
        self.__network.add(Dropout(0.25))
        self.__network.add(Flatten())
        self.__network.add(Dense(128, activation='relu'))
        self.__network.add(Dropout(0.5))
        self.__network.add(Dense(num_classes, activation='softmax'))

        self.__features_extractor = Model(inputs=self.__network.inputs, 
                                          outputs=self.__network.layers[-2].output)
        self.__gclassifier_graph = tf.Graph()
        self.__gaussian_classifier = None

    def fit(self, x_train, y_train, x_validation = None, y_validation = None):
        """
        """
        y_train_m = keras.utils.to_categorical(y_train, self.__num_classes)

        if y_validation is not None:
            y_validation_m = keras.utils.to_categorical(y_validation, self.__num_classes)

        self.__network.compile(loss=keras.losses.categorical_crossentropy,
                               optimizer=keras.optimizers.Adadelta(),
                               metrics=['accuracy'])

        if x_validation is not None:
            self.__network.fit(x_train, y_train_m,
                               batch_size=self.__batch_size,
                               epochs=self.__epochs,
                               verbose=1,
                               validation_data=(x_validation, y_validation_m))
        else:
            self.__network.fit(x_train, y_train_m,
                               batch_size=self.__batch_size,
                               epochs=self.__epochs,
                               verbose=1)


        features_train = self.__features_extractor.predict(x_train).astype(np.float64)

        with self.__gclassifier_graph.as_default():
            kernel = gpflow.kernels.Matern32(input_dim=128) + gpflow.kernels.White(input_dim=128, variance=0.01)
            self.__gaussian_classifier = gpflow.models.SVGP(features_train, y_train, kernel,
                                                            likelihood=gpflow.likelihoods.MultiClass(10),
                                                            Z=features_train[::600].copy(), num_latent=10, whiten=True, q_diag=True)

            self.__gaussian_classifier.kern.white.variance.trainable = False
            self.__gaussian_classifier.feature.trainable = False
            opt = gpflow.train.ScipyOptimizer()
            opt.minimize(self.__gaussian_classifier)

    def predict(self, x, raw_predict=False):
        """
        """
        if raw_predict:
            return self.__network.predict(x)
        else:
            features_x = self.__features_extractor.predict(x).astype(np.float64)
            with self.__gclassifier_graph.as_default():
                return self.__gaussian_classifier.predict_y(features_x)


