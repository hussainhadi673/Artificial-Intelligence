import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import  metrics
from sklearn.linear_model import LogisticRegression

#Arranging Data
data = pd.read_csv('/home/hadi/Downloads/train.csv')
print(data.head())
X = data[['MSSubClass','LotFrontage','LotArea']]
#y = data[['HalfBath']]
y = data[['BsmtFullBath']]

x= np.array(X)
y = np.array(y)
print(x)
print(y)

#Training And  Predicting
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2, random_state=1)
logmodel = LogisticRegression()
logmodel.fit(x_train,y_train)
predict = logmodel.predict(x_test)
print(predict)
acc= metrics.accuracy_score(y_test,predict)
print(acc)
print("Actual Predicted")
for i in range(len(y_test)):
   print(y_test[i],'   ',predict[i])




