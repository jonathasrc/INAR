#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 15:45:09 2018
KNN Classificador
@author: sedetti
"""
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier

iris_datasset = datasets.load_iris()
iris_X = iris_datasset.data
iris_y = iris_datasset.target
print("Iris_Label_Values")
print(np.unique(iris_y))
#Split iris data in train and test data
#A random permutation, to split the data randomly
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
#Create and fit a nearest -neigbor classifier
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


