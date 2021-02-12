from sklearn.cluster import KMeans
import pandas as pd
from matplotlib import pyplot as plt

data = pd.DataFrame({
    'x': [185, 170, 168, 179, 182, 188, 180, 180, 183, 180],
    'y': [72, 56, 60, 68, 72, 77, 71, 70, 84, 88]
})
print(data.head())
model = KMeans(n_clusters=2, random_state=0)
model.fit(data)
labels = model.predict(data)
centriods= model.cluster_centers_
print("centriods", model.cluster_centers_)

colmap={1: 'r', 2: 'b'}
fig = plt.figure(figsize=(5,5))
colors = map(lambda x: colmap[x+1], labels)
color1= list(colors)
plt.scatter(data['x'], data['y'], color= color1, alpha=0.5, edgecolor= 'k')
for idx, centriod in enumerate(centriods):
    plt.scatter(*centriod, color = colmap[idx+1])
plt.xlim(0,200)
plt.ylim(0,100)
plt.show()