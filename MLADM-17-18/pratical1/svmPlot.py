import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC

def svmPlot(X,y,model):
	# create a mesh to plot in
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	h = (x_max / x_min)/100
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
 	np.arange(y_min, y_max, h))

	plt.subplot(1, 1, 1)

	# predict on each mesh grid points.
	Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
	Z = Z.reshape(xx.shape)

	# counter plot with filling color dependent on Z
	plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)

	# scatter plot the real data 
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.xlim(xx.min(), xx.max())
	plt.title('SVC with linear kernel')
	plt.show()
