'''
#Escalonando os dados Min e Max Escalaer fica entre 0 e 1
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        random_state = 1)
scaler = MinMaxScaler(copy=True, feature_range=(0, 1))
scaler.fit(X_train)
print(X_train)
X_train_scaled = scaler.transform(X_train)
print(X_train)
print(X_train_scaled)
print("per-feature_minimun_before_scaling:\n_{}",X_train.min())
print("per-feature_maximun_before_scaling:\n_{}",X_train.max())
print("per-feature_minimun_after_scaling:\n_{}",X_train_scaled.min())
print("per-feature_maximun_after_scaling:\n_{}",X_train_scaled.max())
'''
'''
#Sem escalonar os dados

from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        random_state=0)
svm = SVC(C=100)
svm.fit(X_train, y_train)
print("Test_set_accuracy:_{:.2f}".format(svm.score(X_test, y_test)))
'''
#Escalonando os dados
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
cancer = load_breast_cancer()
print cancer.data[:, :2]
X_train, X_test, y_train, y_test = train_test_split(
        cancer.data, cancer.target,
        random_state=0)

svm = SVC(C=100)
scaler = MinMaxScaler()
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
svm.fit(X_train_scaled, y_train)
print("Teste_de_acuracia:_{:.2f}".format(svm.score(X_test_scaled, y_test)))