from colorsys import yiq_to_rgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("../../resources/Data.csv")
x = dataset.iloc[:,:-1].values
y = dataset.iloc[:,-1].values

y = y.reshape(len(y),1)

## fill missing values

from sklearn.impute import SimpleImputer
iptr = SimpleImputer(missing_values=np.nan, strategy='mean')
x = iptr.fit_transform(x)
y = iptr.fit_transform(y)

y = y.reshape(len(y),1)

#print(x)
#print(y)

## label encoding - not needed

## one-hot encoding - not needed

## feature scaling - not needed

## split into test-train set

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
        

def linReg(x_train,y_train,x_test,y_test):
    linRegressor = LinearRegression()
    linRegressor.fit(x_train, y_train)
    y_pred = linRegressor.predict(x_test)
    return y_pred.reshape(len(y_pred),1)
   
def decTreeReg(x_train, y_train, x_test, y_test):
    dTRegressor = DecisionTreeRegressor()
    dTRegressor.fit(x_train, y_train)
    y_pred = dTRegressor.predict(x_test)
    return y_pred.reshape(len(y_pred),1)
   
def polyReg(x_train, y_train, x_test, y_test):
    pf = PolynomialFeatures(degree = 2)
    x_train = pf.fit_transform(x_train)
    polRegressor = LinearRegression()
    polRegressor.fit(x_train,y_train)
    y_pred = polRegressor.predict(pf.fit_transform(x_test))
    return y_pred.reshape(len(y_pred),1)
   
def svReg(x_train, y_train, x_test, y_test):
    ss_x = StandardScaler()
    x_train_temp = ss_x.fit_transform(x_train[:,:])
    ss_y = StandardScaler()
    y_train_temp = ss_y.fit_transform(y_train[:,:])
    svr = SVR(kernel='rbf')
    svr.fit(x_train_temp, y_train_temp)
    y_pred = svr.predict(ss_x.fit_transform(x_test))
    y_pred = ss_y.inverse_transform(y_pred.reshape(len(y_pred),1))
    return y_pred.reshape(len(y_pred),1)
   
def rfReg(x_train, y_train, x_test, y_test):
    rfRegressor = RandomForestRegressor(n_estimators=10, random_state=0)
    rfRegressor.fit(x_train, y_train)
    y_pred = rfRegressor.predict(x_test)
    return y_pred.reshape(len(y_pred),1)


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x,y,test_size=0.2,random_state=0)

y_linReg = linReg(X_train, Y_train, X_test, Y_test)
y_decTreeReg = decTreeReg(X_train, Y_train, X_test, Y_test)
y_polyReg = polyReg(X_train, Y_train, X_test, Y_test)
y_svReg = svReg(X_train, Y_train, X_test, Y_test)
y_rfReg = rfReg(X_train, Y_train, X_test, Y_test)

y_list = np.concatenate([Y_test, y_linReg, y_decTreeReg, y_polyReg, y_svReg, y_rfReg], axis=1)
print(y_list)

from sklearn.metrics import r2_score
for y_pred in (y_linReg, y_decTreeReg, y_polyReg, y_svReg, y_rfReg):
    print(y_pred,r2_score(Y_test,y_pred))
