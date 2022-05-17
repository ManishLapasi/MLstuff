import numpy as np, pandas as pd, matplotlib.pyplot as plt

dataset = pd.read_csv("/Users/manish/pers/ML/resources/Part 2 - Regression/Section 5 - Multiple Linear Regression/Python/50_Startups.csv")

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

## split into train and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train,y_train)

## remove dummy variable trap - sklearn takes care of this for you
## select best feature using some process like forward progression, backwards elimination, etc. - sklearn takes care of this for you

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

y_pred = regressor.predict(x_test)

np.set_printoptions(precision=2)
concatArr = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)), axis = 1)
print(concatArr)

print(regressor.coef_)


x1 = np.array([[1],[0],[0],[50000],[50000],[50000]])
print(regressor.predict(np.transpose(x1)))