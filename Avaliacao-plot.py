import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

hepatitis = pd.read_csv('hepatitis.data',',')

#hepatitis = hepatitis[hepatitis['Steroid'] != '?']
#hepatitis = hepatitis[hepatitis['Fatigue'] != '?']
hepatitis = hepatitis[hepatitis['Malaise'] != '?']
#hepatitis = hepatitis[hepatitis['Anorexia'] != '?']
#hepatitis = hepatitis[hepatitis['Liver big'] != '?']
#hepatitis = hepatitis[hepatitis['Liver firm'] != '?']
#hepatitis = hepatitis[hepatitis['Spleen palpable'] != '?']
#hepatitis = hepatitis[hepatitis['Spiders'] != '?']
#hepatitis = hepatitis[hepatitis['Ascites'] != '?']
#hepatitis = hepatitis[hepatitis['Varices'] != '?']
hepatitis = hepatitis[hepatitis['Bilirubin'] != '?']
#hepatitis = hepatitis[hepatitis['Alk phosphate'] != '?']
#hepatitis = hepatitis[hepatitis['SGOT'] != '?']
hepatitis = hepatitis[hepatitis['ALBUMIN'] != '?']
#hepatitis = hepatitis[hepatitis['PROTIME'] != '?']


X = hepatitis[['ALBUMIN','Bilirubin']]
y = hepatitis['Die or live']
print X

#Pre-Processamento /escalonando com MinMaxEscaler
scaler = MinMaxScaler(copy= True,feature_range=(0,1))
scaler.fit(X)

X_train_scaled = scaler.transform(X)




h = .02
C = 1.0
svc = svm.SVC(kernel='linear',C=C).fit(X_train_scaled,y)
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X_train_scaled,y)
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X_train_scaled,y)
lin_svc = svm.LinearSVC(C=C).fit(X_train_scaled,y)
x_min, x_max = X_train_scaled[:, 0].min()-1, X_train_scaled[:,0].max()+1
y_min, y_max = X_train_scaled[:, 1].min()-1, X_train_scaled[:,1].max()+1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
titles = ['SVC_with_linear_kernel',
          'LinearSVC_(linear_kernel)',
          'SVC_with_RBF_kernel',
          'SVC_with_polynomial_(degree_3)_kernel']
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    plt.subplot(2, 2, i+1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Sepal_length')
    plt.ylabel('Sepal_width')
    plt.xlim(xx.min(),xx.max())
    plt.ylim(yy.min(),yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
plt.savefig('img/svmdecisionboundary.png')
plt.show()