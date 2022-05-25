import numpy as np, pandas as pd

dataset = pd.read_csv("./Data.csv")
x = dataset.iloc[:,1:-1].values
y = dataset.iloc[:,-1].values
y = y.reshape(len(y),1)

def logReg(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def knn(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski',)
    classifier.fit(x_train, y_train)
    return classifier

def linearSVM(x_train, y_train):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear',random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def nonlinearSVM(x_train, y_train):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def nb(x_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    return classifier

def dt(x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def rf(x_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    classifier = RandomForestClassifier(n_estimators=100, criterion='entropy', random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def evaluateModel(y_true, y_pred):
    from sklearn.metrics import confusion_matrix, accuracy_score
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    print(accuracy_score(y_true,y_pred))
## fill missing values - not needed
## one hot encoding - not needed
## label encoding - needed for y because it is either 2 or 4

from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
y = lb.fit_transform(y)

## split into train test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

## feature scaling - needed for SVM

from sklearn.preprocessing import StandardScaler
ss_x = StandardScaler()
x_train_scaled = ss_x.fit_transform(x_train)
x_test_scaled = ss_x.fit_transform(x_test)
## evaluate models

for model in (logReg, knn, nb, dt, rf):
    classifier = model(x_train, y_train)
    y_pred = classifier.predict(x_test)
    evaluateModel(y_test, y_pred)

for model in (linearSVM, nonlinearSVM):
    classifier = model(x_train_scaled, y_train)
    y_pred = classifier.predict(x_test_scaled)
    evaluateModel(y_test, y_pred)