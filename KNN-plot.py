#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:11:25 2018
plot KNN SLIDE 7
@author: sedetti
"""
import numpy as np
import matplotlib.pyplot as plot
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
n_neighbors = 15
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target
h = .02 #step predicted in the mesh
#Create color maps
cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])
for weights in ['uniform', 'distance']:
    clf = neighbors.KNeighborsClassifier(n_neighbors,weights=weights)
    clf.fit(X,y)
    # Plot the decision boundary
    x_min, x_max = X[:, 0].min() -1, X[:,0].max() +1
    y_min, y_max = X[:, 1].min() -1, X[:,1].max() +1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    # Put the resultin to a color plot
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
