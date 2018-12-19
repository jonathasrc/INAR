'''
#Dispersao dos dados

import matplotlib.pyplot as plt
from pylab import *
import numpy as np
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
plt.scatter(pageSpeeds, purchaseAmount)
plt.savefig('img/plotscatterdata.png')
plt.show()

'''
'''
#Regressao polinomoial
import matplotlib.pyplot as plt
from pylab import *
import numpy as np
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 4))
xp = np.linspace(0, 7, 100)
plt.scatter(x,y)
plt.plot(xp, p4(xp), c='r')
plt.savefig('img/plotpolynomialregression.png')
plt.show()
'''

'''
#ERRO QUADRATICO
import matplotlib.pyplot as plt
from pylab import *CL
import numpy as np
from sklearn.metrics import r2_score
np.random.seed(2)
pageSpeeds = np.random.normal(3.0, 1.0, 1000)
purchaseAmount = np.random.normal(50.0, 10.0, 1000) / pageSpeeds
x = np.array(pageSpeeds)
y = np.array(purchaseAmount)
p4 = np.poly1d(np.polyfit(x, y, 4))
r2 = r2_score(y, p4(x))
print(r2)
'''
'''
#REGRESSAO MULTIVARIADA
#DADOS PANDA PARA COLOCAR EM UM TABELA
import pandas as pd
df = pd.read_excel('datasets/cars.xls')
#print(df.head())
#ATE AQUI
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pandas as pd
df = pd.read_excel('datasets/cars.xls')
scale = StandardScaler()
X = df[['Mileage', 'Cylinder', 'Doors']]
y = df['Price']
X[['Mileage', 'Cylinder', 'Doors']] = scale.fit_transform(X[['Mileage', 'Cylinder', 'Doors']].as_matrix())
#print(X)
est = sm.OLS(y, X).fit()
print(est.summary())
'''
'''
#KNN regressor
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
boston = load_boston()
K = 9
knn = KNeighborsRegressor(n_neighbors=K)
knn.fit(boston.data, boston.target)
print(boston.target[0])
print(knn.predict([boston.data[0]]))
'''
'''
#KNN REGRESSOR PLOTTER
from sklearn.datasets import load_boston
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
boston = load_boston()
K = 9
knn = KNeighborsRegressor(n_neighbors=K)
x, y = boston.data[:50], boston.target[:50]
y_ = knn.fit(x, y).predict(x)
plt.plot(np.linspace(-1, 1, 50), y, label='data', color='black')
plt.plot(np.linspace(-1, 1, 50), y_, label='prediciton', color='red')
plt.legend()
plt.savefig('img/plotknnregression.png')
plt.show()

''