'''mport numpy as np
import pandas as pd
from sklearn import tree
df = pd.read_excel("datasets/jogartenis.xls", sheetname=0)
print(df.head)
d = {'Sim': 1, 'Nao': 0}
df['JogarTenis'] = df['JogarTenis'].map(d)
d = {'Sol': 0, 'Nuvens': 1, 'Chuva': 2}
df['Aspecto'] = df['Aspecto'].map(d)
d = {'Quente': 0, 'Ameno': 1, 'Fresco': 2}
df['Temperatura'] = df['Temperatura'].map(d)
d = {'Normal': 0, 'Elevada': 1}
df['Humidade'] = df['Humidade'].map(d)
d = {'Fraco': 0, 'Forte': 1}
df['Vento'] = df['Vento'].map(d)
print(df.head())
features = list(df.columns[:6])
print(features)
y = df["JogarTenis"]
X = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,y)
print("Score RandomForestn Classifier:")
print(clf.score(X,y))'''

from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pandas import crosstab
np.random.seed(0)
iris = load_iris()
df = pd.DataFrame(iris.data, columns=iris.feature_names)
print(df.head())
df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
print(df.head())
X_train, X_test, Y_train, Y_test = train_test_split(
        iris['data'], iris['target'], random_state=0)
clf = RandomForestClassifier(n_jobs=2, random_state=0)
clf.fit(X_train, Y_train)
predictedClass = clf.predict(X_test)
print('PreditedClasses')
print(predictedClass)
print('Actual_Classes')
print(Y_test)
print(pd.crosstab(Y_test, predictedClass, rownames=['Actual_Speicies'], colnames=['Predicted_Species']))