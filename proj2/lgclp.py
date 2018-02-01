import numpy as np
from sklearn.metrics import accuracy_score

from data.utils import load_data, normalize_adj

class LGCLP:
    def __init__(self, alpha=0.98, iter=20):
        self.f = None
        self.alpha = alpha
        self.iter = iter

    def fit(self, adj_m, Y):
        self.f = np.zeros(Y.shape)
        L = normalize_adj(adj_m)
        for _ in range(self.iter):
            self.f = (self.alpha*L).dot(self.f) + (1 - self.alpha)*Y
        
    def predict(self, x_indexes):
        return np.argmax(self.f[x_indexes], axis=1)

    def evaluate(self, Y_test, test_mask):
        test_index = np.nonzero(test_mask)
        y_true = np.argmax(Y_test[test_mask, :], axis=1)
        y_predicted = self.predict(test_index)
        return accuracy_score(y_true, y_predicted)


if __name__ == "__main__":
    adj, _, Y_train, Y_val, Y_test, train_mask, val_mask, test_mask = load_data("citeseer")
    model = LGCLP()
    model.fit(adj, Y_train)
    print(model.evaluate(Y_test, test_mask))





    