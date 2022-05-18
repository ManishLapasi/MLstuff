from calendar import c
import numpy as np, pandas as pd
from scipy import rand

dataset = pd.read_csv("../../resources/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)

## fill empty values - not needed
## one hot encoding - not needed
## label encoding - not needed

## split into test train test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

## feature scaling
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, metric="minkowski", p=2)
knn.fit(x_train, y_train)

y_pred = knn.predict(ss_x.fit_transform(x_test))
y_pred = y_pred.reshape(len(y_pred),1)
y_list = np.concatenate([y_test, y_pred], axis=1)
print(y_list)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))