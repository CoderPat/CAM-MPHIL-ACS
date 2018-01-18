from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from keras import backend as K

import numpy as np

from gaussiannets import ProbabilisticCNN

if __name__ == "__main__":
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
    print(x_test.shape[0], 'test samples')

    model = ProbabilisticCNN(input_shape, 10, epochs=9)
    model.fit(x_train, y_train)

    y_predict_prob = np.argmax(model.predict(x_test)[0], axis=1)
    y_predict_raw = np.argmax(model.predict(x_test, raw_predict=True), axis=1)

    score_raw = accuracy_score(y_test, y_predict_raw)
    score_gaussian = accuracy_score(y_test, y_predict_prob)

    print('CNN accuracy', score_raw)
    print('ProbabilisticCNN accuracy', score_gaussian)