from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
import numpy as np
from sklearn.metrics import accuracy_score

from data.utils import load_data

class MLP:
    def __init__(self, input_dim=None, classes=None):
        self.model = None
        if input_dim is not None and classes is not None:
            self.build_model(input_shape, output_shape)
    
    def build_model(self, input_dim, classes):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(64, activation='tanh'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes, activation='softmax'))

    def fit(self, X_train, y_train):
        if self.model is None:
            input_dim = np.array(X_train).shape[1]
            classes = np.array(y_train).shape[1]
            self.build_model(input_dim, classes)

        opt = Adam()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt)

        self.model.fit(X_train, y_train,
                       epochs=20,
                       batch_size=64)

    def predict(self, X):
        return np.argmax(self.model.predict(X), axis=1)


if __name__ == "__main__":
    _, features, Y_train, Y_val, Y_test, train_mask, val_mask, test_mask = load_data("citeseer")
    model = MLP()
    model.fit(features[train_mask, :].todense(), Y_train[train_mask, :])

    test_index = np.nonzero(test_mask)
    y_true = np.argmax(Y_test[test_mask, :], axis=1)
    
    y_predicted = model.predict(features[test_mask, :].todense())
    print(accuracy_score(y_true, y_predicted))