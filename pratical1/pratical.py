import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import load_iris, load_digits
from sklearn.metrics import classification_report, accuracy_score
from sklearn.grid_search import GridSearchCV

from svmPlot import svmPlot

TEST_SIZE=15

iris=load_iris()

linear_svms = [SVC(kernel='linear', C=1), SVC(kernel='linear', C=0.001)]
svm_linear = linear_svms[0]

poly_svms = [SVC(kernel='poly', C=1, degree=2, coef0=1)]
rbf_svms = [SVC(kernel='rbf', C=1, gamma=1), SVC(kernel='rbf', C=1, gamma=10), SVC(kernel='rbf', C=10, gamma=10)]

X=iris.data[:,:2]
y=iris.target
indices=np.random.permutation(len(X))

X_train = X[indices[:-TEST_SIZE]]
y_train = y[indices[:-TEST_SIZE]]
X_test = X[indices[-TEST_SIZE:]]
y_test = y[indices[-TEST_SIZE:]]

for svm_l in linear_svms:
    svm_l.fit(X_train, y_train)

for svm_poly in poly_svms:
    svm_poly.fit(X_train, y_train)

for svm_rbf in rbf_svms:
    svm_rbf.fit(X_train, y_train)


y_pred=svm_linear.predict(X_test)

print (classification_report(y_test, y_pred))
print ("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 2))

for svm_l in linear_svms:
    svmPlot(X, y, svm_l)

for svm_poly in poly_svms:
    svmPlot(X, y, svm_poly)

for svm_rbf in rbf_svms:
    svmPlot(X, y, svm_rbf)

g_range = 10. ** np.arange(-3, 1, step=1)
C_range = 10. ** np.arange(-1, 1, step=1)
parameters = [{'gamma' : g_range, 'C': C_range, 'kernel':['rbf']}]

grid = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1)
grid.fit(X_train, y_train)
bestG = grid.best_params_['gamma']
bestC= grid.best_params_['C']
print(np.log2(bestG), np.log2(bestC))

mnist = load_digits()

X=mnist.data
y=mnist.target
indices=np.random.permutation(len(X))

X_train = X[indices[:-TEST_SIZE]]
y_train = y[indices[:-TEST_SIZE]]
X_test = X[indices[-TEST_SIZE:]]
y_test = y[indices[-TEST_SIZE:]]

grid = GridSearchCV(SVC(), parameters, cv=10, n_jobs=-1)
#grid.fit(X_train, y_train)

y_pred = grid.predict(X_test)

print (classification_report(y_test, y_pred))
print ("Overall Accuracy:", round(accuracy_score(y_test, y_pred), 2))




