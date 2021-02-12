from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from matplotlib import pyplot as plt

boston = datasets.load_boston()
print(boston)
x= boston.data
y= boston.target

li = linear_model.LinearRegression()
plt.scatter(x.T[5],y)
plt.show()

x_train, x_test, y_train, y_test= train_test_split(x,y, test_size=1.2)
model = li.fit(x_train,y_train)
predictions = model.predict(x_test)
print("predictions: ", predictions)
print("R^2: ", li.score(x, y))
print("coeff: ", li.coef_)
print("intercept: ", li.intercept_)