import numpy as np, pandas as pd, matplotlib.pyplot as plt

dataset = pd.read_csv("../../resources/Part 2 - Regression/Section 6 - Polynomial Regression/Python/Position_Salaries.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)
print(x)
print(y)

## fill missing values - not needed
## one hot encoding - not needed
## label value encoding - not needed
## split into test-train - not needed
## feature scaling - not needed

from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(x,y)

print(regressor.predict([[7.536]]))

x_grid = np.arange(min(x),max(x),0.1)
y_grid = regressor.predict(x_grid.reshape(len(x_grid),1))
plt.scatter(x,y, color = 'red')
plt.plot(x_grid, y_grid, color = 'blue')
plt.show()