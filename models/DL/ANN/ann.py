from calendar import c
import numpy as np, pandas as pd

dataset = pd.read_csv("../../../resources/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Python/Churn_Modelling.csv")
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,-1].values
print(x[1])

## Label encoding

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

## one hot encoding

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[1])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x[1])
print(x[5])
print(x[6])

## split into test train test

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
print(x_train)

## feature scaling

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
x_train = ss.fit_transform(x_train)

print(x_train)

## initialising ANN

import tensorflow as tf

ann = tf.keras.models.Sequential()

## input layer and first hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

## second hidden layer

ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

## output layer

ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

## compile ANN

ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

## train ann

ann.fit(x_train, y_train, batch_size=32, epochs = 100)

## predict value

y_pred = ann.predict(ss.transform(x_test))
y_pred = (y_pred > 0.5)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

## try predicting for custom value

## 5,15737888,Mitchell,850,Spain,Female,43,2,125510.82,1,1,1,79084.1,0
# Spain, so first 3 variables are 0,0,1
# Female, so 5th variable is 0

x_sample = [0.0, 0.0, 1.0, 850, 0, 43,2,125510.82,1,1,1,79084.1]

print(ann.predict(ss.transform([x_sample])))