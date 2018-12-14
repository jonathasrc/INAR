#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 11:57:59 2018
KNN - exemplo - Slide 06
@author: sedetti
"""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import numpy as np

#KNN Classifier

knn = KNeighborsClassifier(n_neighbors =1)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(iris['data'], iris['target'],
                                                    random_state = 0)
#Aplying the Trainning Data
knn.fit(X_train, y_train)

#Generating a Test Case

X_data_iris_case = np.array([[5, 2.9, 1, 0.2]])
prediction_case = knn.predict(X_data_iris_case)
print(prediction_case)
print(iris['target_names'][prediction_case])

#Predict using the Test Data
y_pred = knn.predict(X_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)