import numpy as np
import pandas as sami
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import neighbors, metrics

data = sami.read_csv('/home/hadi/Desktop/practise sklearn/knn/car.data')
print(data.head())
x = data[['buying','maint','safety']].values
y = data[['class']]
x=np.array(x)
Le = LabelEncoder()
for i in range(len(x[0])):
 x[: , i] = Le.fit_transform(x[: , i])
#print("hashash", x)

label_mapping = {
    'unacc':0,
    'acc':1,
    'good':2,
    'vgood':3
}
y['class'] = y['class'].map(label_mapping)
y = np.array(y)
#print(y)

knn = neighbors.KNeighborsClassifier(n_neighbors=25 , weights='uniform')
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2)
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
accuracy = metrics.accuracy_score(y_test,prediction)
#print("prediction" , prediction)
print("accuracy" , accuracy)
a = 1500
print("actual value ", y[a])
print("predicted value", knn.predict(x)[a])
