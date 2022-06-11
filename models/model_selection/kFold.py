import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 10 - Model Selection _ Boosting/Section 48 - Model Selection/Python/Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

from sklearn.svm import SVC
svc = SVC(kernel='rbf',random_state=0)
svc.fit(x_train, y_train)
y_pred = svc.predict(x_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cm, acc)

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator=svc, X = x_train, y = y_train, cv = 10)
print("Mean accuracy : {:.2f} %".format(accuracies.mean()*100))
print("Mean STD : {:.2f} %".format(accuracies.std()*100))