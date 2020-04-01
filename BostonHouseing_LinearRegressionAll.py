from sklearn import datasets
import pandas as pd

boston = datasets.load_boston()
#print(boston.keys())

df = pd.DataFrame(boston.data, columns = boston.feature_names)
##adding target column to df
df['MEDV'] = boston.target
#print(df.head())

##Creating target and feature arrays
X = df.drop('MEDV', axis = 1).values
y = df['MEDV'].values

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
reg_all = LinearRegression()
reg_all.fit(X_train, y_train)
y_pred = reg_all.predict(X_test)

##SCORE(
print(reg_all.score(X_test, y_test))
print(cross_val_score(reg_all, X, y, cv = 5))