# -*- coding: utf-8 -*-

'''import numpy as np

from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
iris_datasset = datasets.load_iris()
iris_X = iris_datasset.data
iris_y = iris_datasset.target
print("Iris_Label_Values")
print(np.unique(iris_y))
np.random.seed(0)
indices = np.random.permutation(len(iris_X))
print("indices")
print(indices)
iris_X_train = iris_X[indices[:-10]]
iris_y_train = iris_y[indices[:-10]]
iris_X_test = iris_X[indices[-10:]]
iris_y_test = iris_y[indices[-10:]]
print("test_labels")
print(iris_y_test)
knn = KNeighborsClassifier()
knn.fit(iris_X_train, iris_y_train)
KNeighborsClassifier(algorithm='auto', leaf_size=30,
                     metric='minkowski',
                     metric_params=None, n_jobs=1, n_neighbors=5,
                     p=2,
                     weights='uniform')
predictedclasses = knn.predict(iris_X_test)
print("Predicted_Instances")
print(predictedclasses)
print("Iris_Labes_Values")
print(iris_y_test)
classifier_accuaracy = np.mean(predictedclasses == iris_y_test)
print("Classifier_Accuracy")
print(classifier_accuaracy)

import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
n_neighbors = 15
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
h = .02
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)
    x_min, x_max = X[:, 0].min() -1, X[:,0].max() +1
    y_min, y_max = X[:, 1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    predicted = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    predicted = predicted.reshape(xx.shape)
    plot.figure()
    plot.pcolormesh(xx, yy, predicted, cmap=cmap_light)
    plot.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap_bold,
                 edgecolor='k', s=20)
    plot.xlim(xx.min(), xx.max())
   # plot.title("3-Class_Classification_(k_=_%i,_weights_=_'$s')" #Essa linha está causando erro e não deixando salvar o arquivo#
   #            %(n_neighbors, weights))
    plot.savefig("img/plot-knn-%s.jpg"%weights)
plot.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
C = 1.0
svc = svm.SVC(kernel='linear', C=C).fit(X_train, y_train)
y_pred = svc.predict(X_test)
print(y_pred)
print(y_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
C = 1.0
lin_svc = svm.LinearSVC(C=C).fit(X_train, y_train)
y_pred = lin_svc.predict(X_test)
print(y_pred)
print(y_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
C = 1.0
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train, y_train)
y_pred = poly_svc.predict(X_test)
print(y_pred)
print(y_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)'''

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
h = .02
C = 1.0
svc = svm.SVC(kernel='linear',C=C).fit(X,y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X,y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X,y)
lin_svc = svm.LinearSVC(C=C).fit(X,y)
x_min, x_max = X[:, 0].min()-1, X[:,0].max()+1
y_min, y_max = X[:, 1].min()-1, X[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ['SVC_with_linear_kernel',
          'LinearSVC_(linear_kernel)',
          'SVC_with_RBF_kernel',
          'SVC_with_polynomial_(degree_3)_kernel']
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal_length')
    plt.ylabel('Sepal_width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.savefig('img/svmdecisionboundary.png')
plt.show()