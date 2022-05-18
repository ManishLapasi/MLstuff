import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

x = x.reshape(-1,1)
y = y.reshape(-1,1)

## fill empty values - not needed
## one hot encoding - not needed
## split into test train test - not needed
## feature scaling - not needed

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x,y)

import matplotlib.pyplot as plt

def plotter(x,y,y_pred):
    plt.scatter(x,y,color='red')
    plt.plot(x,y_pred,color='blue')
    plt.show()

print(x)
print(x[1][0])
y_pred = regressor.predict(x)

plotter(x,y,y_pred)

#print(regressor.coef_, regressor.intercept_)
#print(regressor.predict([[6.5]]))

from sklearn.preprocessing import PolynomialFeatures
polyF = PolynomialFeatures(degree=2)
xPoly = polyF.fit_transform(x)
polyRegressor = LinearRegression()
polyRegressor.fit(xPoly,y)

y_predPoly = polyRegressor.predict(xPoly)
plotter(x,y,y_predPoly)

inputVal = [[6.5]]

print(regressor.predict(inputVal))
print(polyRegressor.predict(polyF.fit_transform(inputVal)))