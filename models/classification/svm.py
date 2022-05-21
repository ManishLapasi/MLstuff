import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)

## one hot encoding - not needed
## label encoding - not needed

## split into test train set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.25, random_state=0)

## feature scaling
from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)

## not needed on y because it is already between 0 and 1

from sklearn.svm import SVC
classifier = SVC(kernel='linear',random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(ss_x.fit_transform(x_test))
y_pred = y_pred.reshape(len(y_pred),1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))
