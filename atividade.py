#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 16:05:17 2018

@author: sedetti
"""
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn import svm
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer['data'], cancer['target'],
        random_state=0)
C = 1.0
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(
        X_train, y_train)
y_pred = poly_svc.predict(X_test)
print(y_pred)
print(y_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)