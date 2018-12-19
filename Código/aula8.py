'''
SLIDE 08 - REGRESSÃO 
'''
'''
# Coeficiente de Pearson

import pandas as pd
import numpy as np
train_dataset = pd.read_csv('datasets/dadosprecocivic.csv',';')

#print(train_dataset['kilometragem-x'])
X = train_dataset['kilometragem-x']
Y = train_dataset['preco-y']

mediaX = np.mean(X)
mediaY = np.mean(Y)
stdX = np.std(train_dataset['kilometragem-x'], ddof=1)
stdY = np.std(train_dataset['preco-y'], ddof=1)

print("Media_X_%d", mediaX)
print("Media_Y_%d", mediaY)
print("Desvio_Padrao_X_%d", stdX)
print("Desvio_Padrao_Y_%d", stdY)
deviationX = X - mediaX
deviationY = Y - mediaY
sumCovariance = np.sum(deviationX*deviationY)
print("Soma_da_Covariancia_%d", sumCovariance)
N = train_dataset.shape[0]
print(N)
pearsonCorrelation = sumCovariance/((N-1)*stdX*stdY)
print("Correlacao_%4.f", pearsonCorrelation)

beta = pearsonCorrelation * (stdX/stdY)
alpha = mediaY-(beta*mediaX)
print("Beta_%d", beta)
print("Alpha_%d", alpha)
'''

'''
# Regressao
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import matplotlib.pyplot as plt
train_dataset = pd.read_csv('datasets/dadosprecocivic.csv',';')
X = train_dataset['kilometragem-x']
Y = train_dataset['preco-y']
slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
print("(Beta)_Inclinacao_%d",slope)
print("(Alpha)_Intercept_%d",intercept)
print("Coeficiente__Correlacao_%d", r_value)
print("P_Value_%d", p_value)
print("Erro_do_Desvio_Padrao_%d", std_err)

plt.title('Modelo_de_Regressao_Linear_do_Civic')
plt.xlabel('Kilometragem')
plt.ylabel('Preco')
plt.plot(X, Y, 'o', label = 'original_data')
plt.plot(X, intercept + slope*X, 'r', label='fitted_line')
plt.legend()
plt.savefig('img/plotlinearregression.png')
plt.show()


'''
'''
#ERRO QUADRATICO
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
train_dataset = pd.read_csv('datasets/dadosprecocivic.csv',';')
X = train_dataset['kilometragem-x']
Y = train_dataset['preco-y']
slope, intercept, r_value, p_value, std_err = stats.linregress(X,Y)
erroquadratico = r_value ** 2
print("Erro_Quadratico%d", erroquadratico)
'''








