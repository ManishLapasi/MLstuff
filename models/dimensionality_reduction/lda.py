import random
import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 9 - Dimensionality Reduction/Section 44 - Linear Discriminant Analysis (LDA)/Python/Wine.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

print(np.unique(y))

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train = ss_x.fit_transform(x_train)
x_test = ss_x.transform(x_test)

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
lda = LinearDiscriminantAnalysis(n_components=2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

from sklearn.linear_model import LogisticRegression
logReg = LogisticRegression(random_state=0)
logReg.fit(x_train, y_train)

print(logReg.coef_)

y_pred = logReg.predict(x_test)

from sklearn.metrics import accuracy_score, confusion_matrix
cm = confusion_matrix(y_test, y_pred)
acc = accuracy_score(y_test, y_pred)
print(cm,acc)