from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv('/home/hadi/Downloads/quizdata.csv')

data=np.array(data)
print("lenght of input data", len(data))

print(' ')
model = KMeans(n_clusters=2, random_state=0)
model.fit(data)
y = model.predict(data)

print("Input data     belongs to cluster")
for i in range(len(data)):
    print(data[i], '            ', y[i])

cluster_0 = data[y == 0]
#print("cluster 0 :  ", cluster_0)
plt.plot(cluster_0, '*', color='red')


cluster_1 = data[y == 1]
plt.plot(cluster_1 , '+', color='blue')

#print("labels", y)
print("lenth of output ", len(y))

centriods= model.cluster_centers_
plt.plot(centriods, '.', color='black', markersize=10)
plt.legend(['Cluster 0' , 'cluster 1', 'centriods'])

print("centriods", model.cluster_centers_)

plt.show()
