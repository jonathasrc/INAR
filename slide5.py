#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:26:52 2018

@author: sedetti
"""
import numpy as np # Manipular matrizes e matematica de alto nivel
import matplotlib.pyplot as plt # visualizações de dados em gráficos ...
import pandas as pd # Analise de dados
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


""" 
# codigo da pág 42, explicação das bibiliotecas
 # Generate a sequene de integers
x = np.arange(20)
print(x)
# create a second array using sinus 
y = np.sin(x)
print(y)
# the plot function  makes a line chart of one array against nother

plt.plot(x,y, marker="x")
plt.show()
"""
"""
#codigo da pág 45,explicação das bibiliotecas
data  = {'name':["John","Anna","Petter","Linda"],
         'Location': ["New York", "Paris","Berlim","London"],
         'Age': [24,13,53,33]}
data_pandas = pd.DataFrame(data)
print(data_pandas)'                                                                             
"""
# conhecendo as classe pág 54  
iris = load_iris()
#print(iris.keys())
#print(iris['DESCR'][:193] + "\n ...")
#print(iris['target_names'])
#print(type(iris['data']))
#print(iris['data'])
#divindo os dados
X_train, X_test, y_train, y_test = train_test_split(iris['data'],iris['target'],
                                                    random_state =0)
fig, ax = plt.subplots(3,3, figsize = (15,15))
plt.suptitle("Pares de caracteristicas", fontsize = 25)
for i in range (3):
    for j in range(3):
        ax[i,j].scatter(X_train[:,j], X_train[:,i + 1], c=y_train, s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        if i == 2:
            ax[i,j].set_xlabel(iris['feature_names'][j],fontsize = 20)
        if j == 0:
            ax[i,j].set_ylabel(iris['feature_names'][i + 1], fontsize =20)
        if j > i:
            ax[i,j].set_visible(False)
plt.show
            
print("Dados de treinamento:",  X_train.shape)
print("Dados de test :", X_test.shape)
print("Labels de treinament :",  y_train.shape)
print("Labels de test :", y_test.shape)
