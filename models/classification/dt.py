import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

## fill missing values - not needed
## one hot encoding - not needed
## label encoding - not needed

## test train split

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,random_state=0)

## feature scaling - not needed

## build model

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)
y_pred = y_pred.reshape(len(y_pred),1)
y_list = np.concatenate([y_test,y_pred], axis=1)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))