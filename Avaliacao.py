from sklearn import svm
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
Spid ers             5
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

#hepatitis = hepatitis[hepatitis['Steroid'] != '?']
#hepatitis = hepatitis[hepatitis['Fatigue'] != '?']
hepatitis = hepatitis[hepatitis['Malaise'] != '?']
hepatitis = hepatitis[hepatitis['Anorexia'] != '?']
#hepatitis = hepatitis[hepatitis['Liver big'] != '?']
#hepatitis = hepatitis[hepatitis['Liver firm'] != '?']
#hepatitis = hepatitis[hepatitis['Spleen palpable'] != '?']
hepatitis = hepatitis[hepatitis['Spiders'] != '?']
hepatitis = hepatitis[hepatitis['Ascites'] != '?']
#hepatitis = hepatitis[hepatitis['Varices'] != '?']
hepatitis = hepatitis[hepatitis['Bilirubin'] != '?']
#hepatitis = hepatitis[hepatitis['Alk phosphate'] != '?']
#hepatitis = hepatitis[hepatitis['SGOT'] != '?']
hepatitis = hepatitis[hepatitis['ALBUMIN'] != '?']
#hepatitis = hepatitis[hepatitis['PROTIME'] != '?']

#print hepatitis


X_train, X_test, y_train, y_test = train_test_split(
        hepatitis[['Sex','Malaise','Anorexia','Spiders','Ascites','Bilirubin','ALBUMIN']], hepatitis['Die or live'],
        random_state=0)
        

#Pre-Processamento /escalonando com MinMaxEscaler
scaler = MinMaxScaler(copy= True,feature_range=(0,1))
scaler.fit(X_train)
print X_train
X_train_scaled = scaler.transform(X_train)
print X_train_scaled
X_test_scaled = scaler.transform(X_test)


#SVM classifier com kernel linear
svc = svm.SVC(kernel='linear', C=1.0)
svc.fit(X_train,y_train)
svc.fit(X_train_scaled, y_train)
y_pred = svc.predict(X_test)

print "Classe prevista com kernel linear sem esta escalonada "
print y_pred
print "Classe real com kernel linear sem esta escalonada "
print y_test
print np.mean(y_pred == y_test)#outra forma de ver acurracy
print("Teste_de_acuracia kernel linear :_{:.2f}".format(svc.score(X_test,y_test)))
print("Teste_de_acuracia_escalonado:_{:.2f}".format(svc.score(X_test_scaled, y_test)))

print "___________________________________________________________________________________"
#SVM Classifier (LinearSVC(linear Kernel))
lin_svc = svm.LinearSVC(C=1.0).fit(X_train,y_train)
y_pred_lin_svc = lin_svc.predict(X_test)
print y_pred_lin_svc
print y_test
print np.mean(y_pred_lin_svc == y_test)
print("Teste_de_acuracia linear svc:_{:.2f}".format(lin_svc.score(X_test,y_test)))

print "___________________________________________________________________________________"

poly_svc = svm.SVC(kernel= 'poly',degree=3, C=1.0).fit(X_train,y_train)
y_pred_poly_svc = poly_svc.predict(X_test)
print y_pred_poly_svc
print y_test
print "acurracy svc poly"
print np.mean(y_pred_poly_svc == y_test)

#verificando um novo valor para tester vivo ou
X_data_hepatitis_case = np.array([[1,2,2,2,2,0.70,4.0]])
prediction_case = poly_svc.predict(X_data_hepatitis_case)
print(prediction_case)

#print "valores faltantes por linha:"
#print data.apply(num_missing_2, axis = 0) #axis = 0 define que a funcao deve ser aplicada em cada coluna
#print data.apply(num_missing, axis = 1)

D