import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")
x = dataset.iloc[:,1].values
y = dataset.iloc[:,2].values

x = x.reshape(len(x),1)
y = y.reshape(len(y),1)
#print(y)

xact = x
yact = y

## feature scaling - mandatory for SVR / SVM

from sklearn.preprocessing import StandardScaler
ssx = StandardScaler()
x = ssx.fit_transform(x)
ssy = StandardScaler()
y = ssy.fit_transform(y)

#print(x)
#print(y)

## SVR

from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(x,y)

level = 6.5

y_pred = regressor.predict(x)
y_predact = ssy.inverse_transform([y_pred])
print(y_predact)

import matplotlib.pyplot as plt

plt.scatter(xact, yact, color = 'red')
plt.plot(xact, y_predact.reshape(-1,1), color = 'blue')
plt.show()

x_grid = np.arange(min(x),max(x),0.1)
x_grid = x_grid.reshape(len(x_grid),1)
x_grid_act = ssx.inverse_transform(x_grid)
y_grid = regressor.predict(x_grid)
y_grid = y_grid.reshape(len(y_grid),1)
y_grid_act = ssy.inverse_transform(y_grid)
plt.scatter(xact, yact, color='red')
plt.plot(x_grid_act, y_grid_act, color = 'blue')
plt.show()