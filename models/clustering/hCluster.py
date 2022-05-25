import numpy as np, pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("../../resources/Part 4 - Clustering/Section 24 - K-Means Clustering/Python/Mall_Customers.csv")
x = dataset.iloc[:,[3,4]].values

## dendogram

import scipy.cluster.hierarchy as sch
ddgram = sch.dendrogram(sch.linkage(x,method='ward'))
plt.show()

from sklearn.cluster import AgglomerativeClustering
hCluster = AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
y_cluster_pred = hCluster.fit_predict(x)

print(y_cluster_pred)

colormap = ['blue','green','yellow','cyan','magenta']
for i in range(5):
    plt.scatter(x[y_cluster_pred==i,0],x[y_cluster_pred==i,1],color=colormap[i],label='Cluster'+str(i+1))
plt.legend()
plt.show()