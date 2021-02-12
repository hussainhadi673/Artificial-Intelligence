from sklearn import datasets, svm, metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris=datasets.load_iris()
x= iris.data #feature
y= iris.target #label
classes= iris.target_names

#print(x[0])
#print(y)
#print(x[0])
#print(z)
#print(x.shape)
#print(y.shape)


x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
model = svm.SVC()
model.fit(x_train,y_train)
prediction= model.predict(x_test)
acc= metrics.accuracy_score(y_test,prediction)
print(acc)