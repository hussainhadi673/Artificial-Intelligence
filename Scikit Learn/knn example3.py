from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.DataFrame({
    'STR': [85.4, 88.8, 123, 137, 56.8, 74.4, 78.6, 85.8],
    'AGI' : [59, 96, 39, 49, 62.2, 90.2, 63.6, 48.6],
    'INT': [69, 58.8, 53.4, 64.4, 91.4, 51, 102.2, 54]
})

#print(data.head(8))

init = np.array([[85.4, 59, 69],
                     [88.8, 96, 58.8],
                     [123, 39, 53.4],
                     [137, 49, 64.4]],
                    np.float64)

data = np.array(data)

model = KMeans(n_clusters=4 , init=init )
model.fit(data)
labels = model.predict(data)
centriods= model.cluster_centers_
print(" final centriods", model.cluster_centers_)

print("Input data                 belongs to cluster")
for i in range(len(data)):
    print(data[i], '            ', labels[i])

cluster1 =  data[labels == 0]
cluster2 =  data[labels == 1]
cluster3 =  data[labels == 2]
cluster4 =  data[labels == 3]



