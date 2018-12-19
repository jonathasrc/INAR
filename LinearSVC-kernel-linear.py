#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 17 17:07:24 2018
LinearSVC(linera kernel)
RBF
@author: sedetti
"""

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