"""
Usage:
    mnist_test.py [options]

Options:
    --save-path FILE 
    --load-path FILE
"""
from keras.datasets import mnist
from sklearn.metrics import accuracy_score
from keras import backend as K
import keras

import numpy as np
from matplotlib import pyplot as plt
import random

from docopt import docopt

from gaussiannets import ProbabilisticCNN

def print_digit(digit_data, digit, prob_raw, prob_unc, std):
    two_d = (np.reshape(digit_data, (28, 28)) * 255).astype(np.uint8)
    plt.title('digit %d with prob_raw: %.1f%% and prob_unc: %.1f%% +- %.2f' % (digit, prob_raw * 100, prob_unc * 100, 2*std*100))
    plt.imshow(two_d, cmap='gray')
    plt.show()

if __name__ == "__main__":
    args = docopt(__doc__)
    save_path = args.get('--save-path')
    load_path = args.get('--load-path')

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

    y_train_m = keras.utils.to_categorical(y_train, 10)

    model = ProbabilisticCNN(input_shape, 10, epochs=9)

    class_prob_median, class_prob_std = model.predict(x_test)
    class_prob_raw = model.predict(x_test, raw_predict=True)

    y_predict_unc = np.argmax(class_prob_median, axis=1)
    y_predict_raw = np.argmax(class_prob_raw, axis=1)

    score_raw = accuracy_score(y_test, y_predict_raw)
    score_gaussian = accuracy_score(y_test, y_predict_unc)

    print('CNN accuracy', score_raw)
    print('ProbabilisticCNN accuracy', score_gaussian)

    worst_results = np.argsort(np.max(class_prob_median, axis=1))[:100]
    while True:
        random_digit = random.choice(worst_results)
        best_class = np.argmax(class_prob_median[random_digit])
        print_digit(x_test[random_digit], y_predict_unc[random_digit], np.max(class_prob_raw[random_digit]), 
                    class_prob_median[random_digit, best_class], class_prob_std[random_digit, best_class])




