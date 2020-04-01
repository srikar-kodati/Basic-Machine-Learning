from sklearn import datasets
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

boston = datasets.load_boston()
#print(boston.keys())

df = pd.DataFrame(boston.data, columns = boston.feature_names)
##adding target column to df
df['MEDV'] = boston.target
#print(df.head())

##Creating target and feature arrays
X = df.drop('MEDV', axis = 1).values
y = df['MEDV'].values

##using single feature rooms (slicing from df)
X_rooms = X[:,5]
#print(type(X_rooms))
#print(type(y))

x = y.reshape(-1,1)
X_rooms = X_rooms.reshape(-1,1)  

plt.scatter(X_rooms, y)
plt.ylabel('Value of house/1000 ($)')
plt.xlabel('Number of Rooms')
plt.show()

from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X_rooms, y)

prediction_space = np.linspace(min(X_rooms), max(X_rooms)).reshape(-1,1)

plt.scatter(X_rooms, y, c = 'b')
plt.plot(prediction_space, reg.predict(prediction_space), color = 'black', linewidth = 3)
plt.show()
