from sklearn.datasets import load_breast_cancer
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
import numpy as np

cancer = datasets.load_breast_cancer()
X = cancer.data[:,:2]
y = cancer.target

h = .02
C = 1.0

X_train, X_test, y_train, y_test = train_test_split(
    X,y,
     random_state = 0)
svc = svm.SVC(kernel='linear', C=C).fit(X_train,y_train)
lin_svc = svm.LinearSVC(C=C).fit(X_train,y_train)

y_pred = svc.predict(X_test)
y_pred_lin = svc.predict(X_test)


print(y_pred)
print("--------------")
print(y_test)

classifier_score = np.mean(y_pred == y_test)
classifier_score_lin = np.mean(y_pred_lin == y_test)
print("SVM com kernel linear: ",classifier_score)
print("Linear SVM : ",classifier_score_lin)

 

