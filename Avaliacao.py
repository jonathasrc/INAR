from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
def num_missing (x):
    return sum (x.isnull())

def num_missing_2(x):
    return sum(x == "?")

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
print hepatitis.apply(num_missing_2, axis = 0)

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

print hepatitis


X_train, X_test, y_train, y_test = train_test_split(
        hepatitis[['Sex','Malaise','Anorexia','Spiders','Ascites','Bilirubin','ALBUMIN']], hepatitis['Die or live'],
        random_state=0)
        

#Pre-Processamento /escalonando com MinMaxEscaler
scaler = MinMaxScaler(copy= True,feature_range=(0,1))
scaler.fit(X_train)
#print X_train
X_train_scaled = scaler.transform(X_train)
#print X_train_scaled
X_test_scaled = scaler.transform(X_test)

svm = SVC(C=100)
svm.fit(X_train,y_train)
svm.fit(X_train_scaled, y_train)

print("Teste_de_acuracia:_{:.2f}".format(svm.score(X_test,y_test)))
print("Teste_de_acuracia_escalonado:_{:.2f}".format(svm.score(X_test_scaled, y_test)))

'''
#print "valores faltantes por linha:"
#print data.apply(num_missing_2, axis = 0) #axis = 0 define que a funcao deve ser aplicada em cada coluna
#print data.apply(num_missing, axis = 1)

'''