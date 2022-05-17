import pandas as pd, numpy as np, matplotlib.pyplot as plt

dataset = pd.read_csv("../../resources/Part 2 - Regression/Section 4 - Simple Linear Regression/Python/Salary_Data.csv")

x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(-1,1)

## reshape is used when you want to convert 1D array to nested 1D array for processing properly

#print(x)
#print(y)

## fill for missing values? - no missing values, but let's fill in just in case.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
x[:,:] = imputer.fit_transform(x[:,:])
y[:,:] = imputer.fit_transform(y[:,:])

#print(x)

## no nonstring data - no need for onehot encoding

## dependent variable is numerical, no need for label encoding

## train test split

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
#print(x_train)
#print(x_test)

## feature scaling - not needed when there is only one variable

#from sklearn.preprocessing import StandardScaler
#ss = StandardScaler()
#x_train[:] = ss.fit_transform(x_train[:])
#x_test[:] = ss.fit_transform(x_test[:])
#y_train[:] = ss.fit_transform(y_train[:])
#y_test[:] = ss.fit_transform(y_test[:])

#print(x_train)
#print(y_train)

## make the linear regression model

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

## predict the outcome

y_predicted = regressor.predict(x_test)

#print(y_predicted)
#print(y_test)

plt.scatter(x_train, y_train, color = "red")
plt.plot(x_train, regressor.predict(x_train), color = "blue")
plt.show()

plt.scatter(x_test, y_test, color = "red")
plt.plot(x_test, y_predicted, color = "blue")
plt.show()

#print(regressor.predict([[12]]))

print(regressor.coef_,regressor.intercept_)

import lr

(a,b) = lr.retExpLR(x_train, y_train)
print(a,b)

print(x_train, y_train)

a = lr.retExpMLR(x_train, y_train)
print(a)