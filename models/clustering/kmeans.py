import numpy as np, pandas as pd

dataset = pd.read_csv("../../resources/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv")
#x = dataset.iloc[:,1:-1].values
x = dataset.iloc[:,[3,4]].values

## fill nan values - not needed

## one hot encoding 
## this is commented out - we will try clustering with 2 variables first

'''
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[0])],remainder='passthrough')
x = np.array(ct.fit_transform(x))
print(x)
'''

## label encoding - not needed

## feature scaling - needed? since we are clustering
## not needed for this implementation

from sklearn.cluster import KMeans
wss = []
for num_clusters in range(1,10):
    km = KMeans(n_clusters=num_clusters,init='k-means++',random_state=0)
    km.fit(x)
    wss.append(km.inertia_)

import matplotlib.pyplot as plt
plt.plot(range(1,10),wss)
plt.show()

## from the graph, num_clusters is at 5

kmeans = KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans = kmeans.fit_predict(x)

colormap = ['cyan','blue','green','yellow','purple']
for i in range(5):
    plt.scatter(x[y_kmeans==i,0],x[y_kmeans==i,1],s=100,color=colormap[i-1],,label='Cluster'+str(i+1))
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,color='red')
plt.show()