import numpy as np
import pandas as pd
from sklearn import neighbors, svm, metrics
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

train = pd.read_csv('/home/hadi/Desktop/advanceTopic/assignment#6/traindataassignment6.csv')
test  = pd.read_csv('/home/hadi/Desktop/advanceTopic/assignment#6/testdataassignment6.csv')

print("printing 5 entries of csv to see how data looks like : ")
print(train.head())

x = train[['F1' , 'F2', 'F3', 'F4']]
x=np.array(x)
#print(x)

y= train.F5
y=np.array(y)
#print(y)

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=0.2)
model = svm.SVC()
model.fit(x_train,y_train)
prediction= model.predict(x_test)
acc= metrics.accuracy_score(y_test,prediction)
print("accuracy acheived using SVM",  acc)
test = np.array(test)
print("lenght of test data", len(test))
test_data_svm = model.predict(test)
print("predicted output of test dataset using SVM", test_data_svm)
print("printing each label against input to make it user friendly")
for i in range(len(test_data_svm)):
    print(test[i],test_data_svm[i])




