import enum
import numpy as np, pandas as pd
from scipy import rand

dataset = pd.read_csv("../../resources/Part 3 - Classification/Section 14 - Logistic Regression/Python/Social_Network_Ads.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(x_train, y_train)

y_pred = classifier.predict(ss_x.fit_transform(x_test))
y_pred = y_pred.reshape(len(y_pred),1)
y_list = np.concatenate([y_test, y_pred], axis=1)
print(y_list)

print(x_test)

x_val = np.array([[20] ,[30000]])
x_val = x_val.transpose()
print(x_val)
print(x_test[0])

print(classifier.predict(ss_x.transform(x_val)))

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
score = accuracy_score(y_test, y_pred)
print(score)
