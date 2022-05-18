import numpy as np, pandas as pd, matplotlib.pyplot as plt

dataset = pd.read_csv("../../resources/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

## fill for nan values
'''
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0.0, strategy='mean')
x[:,0:2] = imp.fit_transform(x[:,0:2])
y[:,:] = imp.fit_transform(y[:,:])

print(x)
print(y)
'''

## one-hot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [3])], remainder='passthrough')
#x[:,3] = LabelEncoder().fit_transform(x[:,3])
x = np.array(ct.fit_transform(x))

print(x)

## append ones to perform MLR

##x = np.append(arr = np.ones((np.size(x,axis = 0),1)).astype(int), values = x, axis = 1)

##print(x)

## split into train and test set



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train,y_train)

import lr

print(np.shape(x_train))

a = lr.retExpMLR(x_train.astype(float), y_train.astype(float))
print(a)
print(y_test)
np.set_printoptions(precision=2)
y_pred2 = np.matmul(x_test,a)
print(y_pred2)

x1 = [[1],[0],[0],[50000],[50000],[50000]]
print(np.matmul(a,x1))
