from sklearn import datasets
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

boston = datasets.load_boston()
df = pd.DataFrame(boston.data, columns = boston.feature_names)
df['MEDV'] = boston.target

X = df.drop('MEDV', axis = 1).values
y = df['MEDV'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 42)
ridge = Ridge(alpha = 0.1, normalize = True)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
print(ridge.score(X_test, y_test))

from sklearn.linear_model import Lasso

df_columns = boston.feature_names

lasso = Lasso(alpha = 0.1, normalize = True)
lasso_coef = lasso.fit(X_train, y_train).coef_
print(lasso_coef)

plt.plot(range(len(df_columns)), lasso_coef)
plt.xticks(range(len(df_columns)), df_columns, rotation = 60)
plt.margins(0.02)
plt.show()

lasso_pred = lasso.predict(X_test)
print(lasso.score(X_test, y_test))