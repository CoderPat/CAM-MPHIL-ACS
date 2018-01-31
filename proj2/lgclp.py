import numpy as np
from sklearn.metrics import accuracy_score

from data.utils import load_data, normalize_adj

class LGCLP:
    def __init__(self, alpha=0.8, iter=20):
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


if __name__ == "__main__":
    adj, _, Y_train, Y_val, Y_test, train_mask, val_mask, test_mask = load_data("citeseer")
    model = LGCLP()
    model.fit(adj, Y_train)

    test_index = np.nonzero(test_mask)
    y_true = np.argmax(Y_test[test_mask, :], axis=1)
    
    y_predicted = model.predict(test_index)
    print(accuracy_score(y_true, y_predicted))





    