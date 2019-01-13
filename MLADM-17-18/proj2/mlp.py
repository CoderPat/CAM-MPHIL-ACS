from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

import numpy as np
import scipy.sparse as sp
from sklearn.metrics import accuracy_score

from data.utils import load_data

def normalize_features(features):
    """Row-normalize feature matrix and convert to tuple representation"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    return features.todense()

class MLP:
    def __init__(self, input_dim=None, classes=None):
        self.model = None
        if input_dim is not None and classes is not None:
            self.build_model(input_shape, output_shape)
    
    def build_model(self, input_dim, classes):
        self.model = Sequential()
        self.model.add(Dense(64, input_dim=input_dim, activation='relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes, activation='softmax'))

    def fit(self, X_train, y_train, X_validation=None, Y_validation=None):
        normalized_features = normalize_features(X_train)
        if self.model is None:
            input_dim = normalized_features.shape[1]
            classes = y_train.shape[1]
            self.build_model(input_dim, classes)

        opt = Adam()
        self.model.compile(loss='categorical_crossentropy',
                           optimizer=opt,
                           metrics=['acc'])

        if X_validation is not None and Y_validation is not None:
            early_stopping = EarlyStopping(monitor='val_acc', patience=10)
            self.model.fit(normalized_features, y_train,
                           epochs=100,
                           batch_size=64,
                           validation_data=(X_validation, Y_validation),
                           callbacks=[early_stopping])
        else:
            self.model.fit(normalized_features, y_train,
                           epochs=100,
                           batch_size=64)



    def predict(self, X):
        normalized_features = normalize_features(X)
        return np.argmax(self.model.predict(normalized_features), axis=1)

    def evaluate(self, X_test, Y_test):
        normalized_features = normalize_features(X_test)
        _, accuracy = self.model.evaluate(normalized_features, Y_test)
        return accuracy


if __name__ == "__main__":
    _, features, Y_train, Y_val, Y_test, train_mask, val_mask, test_mask = load_data("citeseer")
    model = MLP()
    model.fit(features[train_mask, :], Y_train[train_mask, :])
    print(model.evaluate(features[test_mask, :], Y_test[test_mask, :]))
