#REGRESSAO MULTIVARIADA
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import pandas as pd

'''
Die or live         0
Age                 0
Sex                 0
Steroid             1
Antivirals          0
Fatigue             1
Malaise             1
Anorexia            1
Liver big          10
Liver firm         11
Spleen palpable     5
Spiders             5
Ascites             5
Varices             5
Bilirubin           6
Alk phosphate      29
SGOT                4
ALBUMIN            16
PROTIME            67
HISTOLOGY           0
'''
hepatitis = pd.read_csv('hepatitis.data',',')


hepatitis = hepatitis[hepatitis['Steroid'] != '?']
hepatitis = hepatitis[hepatitis['Fatigue'] != '?']
hepatitis = hepatitis[hepatitis['Malaise'] != '?']
hepatitis = hepatitis[hepatitis['Anorexia'] != '?']
hepatitis = hepatitis[hepatitis['Liver big'] != '?']
hepatitis = hepatitis[hepatitis['Liver firm'] != '?']
hepatitis = hepatitis[hepatitis['Spleen palpable'] != '?']
hepatitis = hepatitis[hepatitis['Spiders'] != '?']
hepatitis = hepatitis[hepatitis['Ascites'] != '?']
hepatitis = hepatitis[hepatitis['Varices'] != '?']
hepatitis = hepatitis[hepatitis['Bilirubin'] != '?']
hepatitis = hepatitis[hepatitis['Alk phosphate'] != '?']
hepatitis = hepatitis[hepatitis['SGOT'] != '?']
hepatitis = hepatitis[hepatitis['ALBUMIN'] != '?']
hepatitis = hepatitis[hepatitis['PROTIME'] != '?']

scale = StandardScaler()
X = hepatitis[['Age','Sex','Antivirals']]
y = hepatitis['Die or live']
X[['Malaise', 'Sex', 'ALBUMIN']] = scale.fit_transform(X[['Malaise', 'Sex', 'ALBUMIN']].as_matrix())
#print(X)
est = sm.OLS(y, X).fit()
print(est.summary())