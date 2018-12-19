'''import numpy as np
x = np.array([[1,2,3],[4,5,6]])
print(x)

from scipy import sparse
import numpy as np
eye = np.eye(4)
print ("Numpy_array:\n%s" % eye)

import numpy as np
import matplotlib.pyplot as plt
x = np.arange(20)
y = np.sin (x)
plt.plot(x,y,marker="x")
plt.show()

import pandas as pd
data = {'Name': ["John", "Ana", "Peter", "Linda"],
        'Location':["New_York", "Paris", "Berlin", "London"],
        'Age':[24, 13, 53, 33]}
data_pandas = pd.DataFrame(data)
print(data_pandas)

from sklearn.datasets import load_iris
iris = load_iris()
print (type(iris))

#Retorna o seguinte erro: <class 'sklearn.utils.Bunch'>#

from sklearn.datasets import load_iris
iris = load_iris()
print (iris.keys())
print (iris ['DESCR'][:193]+"\n...")

from sklearn.datasets import load_iris
iris = load_iris()
print(iris['target_names'])

from sklearn.datasets import load_iris
iris = load_iris()
print(iris['feature_names'])

from sklearn.datasets import load_iris
iris = load_iris()
print(type(iris['data']))
print (iris['data'])

from sklearn.datasets import load_iris
iris = load_iris()
numpy.ndarray 
print(iris['target'])

from sklearn.datasets import load_iris #Entender melhor esses comandos#
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
fig, ax = plt.subplots(3, 3, figsize=(15, 15))
plt.suptitle("Pares_de_Caracteristicas", fontsize=25)
for i in range(3):
    for j in range(3):
        ax[i,j].scatter(X_train[:,j], X_train[:, i + 1],
          c=y_train, s=60)
        ax[i,j].set_xticks(())
        ax[i,j].set_yticks(())
        if i==2:
            ax[i,j].set_xlabel(iris['feature_names'][j],
              fontsize=20)
        if i==0:
            ax[i,j].set_ylabel(iris['feature_names'][i+1],
              fontsize=20)
        if j>i:
            ax[i,j].set_visible(False)
plt.show()

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors=1)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
knn.fit(X_train,y_train)
X_data_iris_case = np.array([[3, 2.9, 1, 0.2]])
prediction_case = knn.predict (X_data_iris_case)
print(prediction_case)
print(iris['target_names'][prediction_case])'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
knn = KNeighborsClassifier(n_neighbors=1)
iris = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
        iris['data'], iris['target'],
        random_state=0)
knn.fit(X_train,y_train)
y_pred = knn.predict(X_test)
classifier_score = np.mean(y_pred == y_test)
print(classifier_score)




































