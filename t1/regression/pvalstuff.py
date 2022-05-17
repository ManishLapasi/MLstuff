from cmath import inf
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

## append ones to perform MLR

##x = np.append(arr = np.ones((np.size(x,axis = 0),1)).astype(int), values = x, axis = 1)

##print(x)

## split into train and test set



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)
print(x_train,y_train)

from scipy import stats

p_cutoff = 0.05
flag = 1
p_list = []

while flag == 1:
    p_list = []
    max_ind = 0
    max_pval = 0
    for i in range(len(x[0])):
        p_val = stats.ttest_ind(x[:,i],y)[1]
        p_list.append(p_val)
        if p_val>max_pval:
            max_pval = p_val
            max_ind = i
    print("max pval ", max_pval, " at column ",max_ind)
    print(p_list)
    if max_pval > p_cutoff:
        print("deleting column ", max_ind)
        x = np.delete(x,max_ind,axis=1)
        flag = 1
    else:
        flag = 0